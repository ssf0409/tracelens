"""Tests for the paired task bootstrap behind `tracelens compare` (issue #28).

Expectations are derived by hand or from an independent computation, per the
statistical contract's run-versus-run section.
"""

from __future__ import annotations

import json
import math
from itertools import product

import numpy as np
import pytest

from tracelens.core.outcome import Outcome
from tracelens.core.provenance import (
    CandidateSpec,
    ComponentIdentity,
    MeasurementSetup,
    RunnerSettings,
    RunProvenance,
)
from tracelens.core.trial import Trial, TrialBatch, TrialStatus
from tracelens.statistics.inference import bootstrap_difference_ci
from tracelens.statistics.run_comparison import (
    ComparisonError,
    Direction,
    MetricSelector,
    PairedEffect,
    RunComparison,
    Verdict,
    compare_runs,
    decide,
    excludes_zero,
    paired_task_effect,
)

# --- builders ------------------------------------------------------------------


def _trial(
    task_id: str,
    run_index: int,
    *,
    passed: bool | None = None,
    score: float | None = None,
    metrics: dict[str, float] | None = None,
    status: TrialStatus = TrialStatus.COMPLETED,
    grader: str = "g",
    extra: dict[str, tuple[bool, float]] | None = None,
) -> Trial:
    """A trial with one outcome for ``grader`` (and optional extra graders)."""
    trial = Trial(task_id=task_id, run_index=run_index, status=status)
    if passed is not None:
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id, grader_id=grader, passed=passed,
            score=(1.0 if passed else 0.0) if score is None else score,
            metrics=metrics or {},
        ))
    for other, (other_passed, other_score) in (extra or {}).items():
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id, grader_id=other, passed=other_passed, score=other_score,
        ))
    return trial


def _provenance(
    task_ids: list[str],
    *,
    hashes: dict[str, str] | None = None,
    graders: tuple[str, ...] = ("g",),
    adapter_version: str | None = None,
    run_id: str = "run",
) -> RunProvenance:
    task_hashes = {t: (hashes or {}).get(t, "h-" + t) for t in task_ids}
    return RunProvenance(
        run_id=run_id,
        measurement=MeasurementSetup(
            eval_set_hash="".join(sorted(task_hashes.values()))[:64].ljust(64, "0"),
            task_hashes=task_hashes,
            graders=[ComponentIdentity(class_path=f"x.{g}", name=g) for g in graders],
            runner=RunnerSettings(
                num_runs=1, max_concurrency=1, timeout_seconds=1.0, max_infra_retries=0
            ),
        ),
        candidate=CandidateSpec(
            adapter=ComponentIdentity(class_path="x.Adapter", version=adapter_version)
        ),
    )


def _pass_batch(
    passes: dict[str, list[bool]],
    *,
    provenance: bool = True,
    hashes: dict[str, str] | None = None,
    graders: tuple[str, ...] = ("g",),
    run_id: str = "run",
) -> TrialBatch:
    """A batch of pass/fail trials, ``{task_id: [passed per run]}``."""
    batch = TrialBatch()
    for task_id, results in passes.items():
        for i, passed in enumerate(results):
            batch.add_trial(_trial(task_id, i, passed=passed))
    if provenance:
        batch.provenance = _provenance(
            sorted(passes), hashes=hashes, graders=graders, run_id=run_id
        )
    return batch


# --- MetricSelector --------------------------------------------------------------


class TestMetricSelector:
    def test_builtins_are_higher_is_better(self):
        assert MetricSelector.parse() == MetricSelector("pass_rate", Direction.HIGHER)
        assert MetricSelector.parse("mean_score", grader="g") == MetricSelector(
            "mean_score", Direction.HIGHER, "g"
        )
        with pytest.raises(ComparisonError, match="always higher-is-better"):
            MetricSelector.parse("pass_rate", direction="lower")

    def test_custom_metric_names_its_grader_and_direction(self):
        selector = MetricSelector.parse("budget.latency_ms", direction="lower")
        assert selector == MetricSelector("budget.latency_ms", Direction.LOWER, "budget")
        assert selector.sign == -1.0
        assert MetricSelector.parse("g.m").direction is Direction.HIGHER

    @pytest.mark.parametrize(
        ("metric", "direction", "grader", "fragment"),
        [
            ("latency", None, None, "unknown metric 'latency'"),
            ("g.", None, None, "unknown metric"),
            (".m", None, None, "unknown metric"),
            ("g.m", "sideways", None, "--direction must be"),
            ("g.m", None, "other", "conflicts with the grader"),
        ],
    )
    def test_rejects_bad_selections(self, metric, direction, grader, fragment):
        with pytest.raises(ComparisonError, match=fragment):
            MetricSelector.parse(metric, direction, grader)

    def test_values_follow_the_trial_validity_rules(self):
        pass_rate = MetricSelector.parse("pass_rate")
        by_grader = MetricSelector.parse("pass_rate", grader="g")
        score = MetricSelector.parse("mean_score")
        latency = MetricSelector.parse("g.latency", direction="lower")

        assert pass_rate.value(_trial("t", 0, passed=True)) == 1.0
        assert pass_rate.value(_trial("t", 0, passed=False)) == 0.0
        # a timeout with no outcome is a failure, also under --grader
        timeout = _trial("t", 0, status=TrialStatus.TIMEOUT)
        assert pass_rate.value(timeout) == 0.0 and by_grader.value(timeout) == 0.0
        # graded by another grader only: no value for this one
        other_only = _trial("t", 0, passed=True, grader="h")
        assert by_grader.value(other_only) is None
        # harness failures and never-run trials have no value
        assert pass_rate.value(_trial("t", 0, passed=True, status=TrialStatus.INFRA_ERROR)) is None
        assert pass_rate.value(_trial("t", 0, status=TrialStatus.PENDING)) is None
        # mean_score: trial aggregate, or the grader's own score, or nothing
        assert score.value(_trial("t", 0, passed=True, score=0.4, extra={"h": (True, 0.8)})) == pytest.approx(0.6)
        assert MetricSelector.parse("mean_score", grader="h").value(
            _trial("t", 0, passed=True, score=0.4, extra={"h": (True, 0.8)})
        ) == 0.8
        assert score.value(timeout) is None
        # outcome metrics: missing or non-finite means no value
        assert latency.value(_trial("t", 0, passed=True, metrics={"latency": 120.0})) == 120.0
        assert latency.value(_trial("t", 0, passed=True)) is None
        assert latency.value(_trial("t", 0, passed=True, metrics={"latency": math.nan})) is None


# --- paired_task_effect ----------------------------------------------------------


class TestPairedTaskEffect:
    def test_hand_derived_delta_and_exact_sign_flip_p_value(self):
        # d = (0.2, -0.1, 0.4): mean 1/6. Of the 8 sign assignments of
        # (0.2, 0.1, 0.4), |sum| >= 0.5 for +++, +-+, -+-, ---: p = 4/8.
        effect = paired_task_effect([0.2, -0.1, 0.4], n_bootstrap=1000, seed=1)
        assert effect.tasks == 3
        assert effect.delta == pytest.approx(0.5 / 3)
        assert effect.p_value == 0.5 and effect.p_value_exact
        assert effect.ci_lower is not None and effect.ci_upper is not None
        assert effect.ci_lower <= effect.delta <= effect.ci_upper

    def test_exact_p_value_matches_full_enumeration(self):
        diffs = [0.31, -0.07, 0.12, 0.25, -0.2, 0.05, 0.4]
        delta = sum(diffs) / len(diffs)
        count = sum(
            1 for signs in product((-1, 1), repeat=len(diffs))
            if abs(sum(s * abs(d) for s, d in zip(signs, diffs, strict=True)) / len(diffs))
            >= abs(delta) - 1e-12
        )
        effect = paired_task_effect(diffs, n_bootstrap=1000, seed=0)
        assert effect.p_value_exact and effect.p_value == pytest.approx(count / 2 ** len(diffs))

    def test_random_sign_flip_when_enumeration_is_too_large(self):
        diffs = [0.1] * 13  # 2^13 > 500 draws
        effect = paired_task_effect(diffs, n_bootstrap=500, seed=0)
        assert not effect.p_value_exact
        assert effect.p_value is not None and 0.0 < effect.p_value <= 1.0
        # all differences positive: only the all-plus assignment is as extreme
        assert effect.p_value == pytest.approx(1 / 501, abs=0.01)

    def test_bootstrap_interval_is_the_percentile_of_task_resample_means(self):
        diffs = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        effect = paired_task_effect(diffs.tolist(), confidence=0.9, n_bootstrap=2000, seed=7)
        rng = np.random.default_rng(7)
        index = rng.integers(0, 5, size=(2000, 5))
        means = diffs[index].mean(axis=1)
        assert effect.ci_lower == pytest.approx(float(np.percentile(means, 5)))
        assert effect.ci_upper == pytest.approx(float(np.percentile(means, 95)))

    def test_seed_reproduces_and_task_order_does_not_matter(self):
        a = paired_task_effect([0.3, -0.1, 0.05, 0.2], seed=3, n_bootstrap=500)
        b = paired_task_effect([0.05, 0.2, 0.3, -0.1], seed=3, n_bootstrap=500)
        assert a == b
        assert paired_task_effect([0.3, -0.1, 0.05, 0.2], seed=4, n_bootstrap=500) != a

    def test_fewer_than_two_tasks_has_no_interval(self):
        assert paired_task_effect([]).delta is None
        one = paired_task_effect([0.4])
        assert one.delta == 0.4 and one.ci_lower is None and one.p_value is None

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError, match="confidence"):
            paired_task_effect([0.1, 0.2], confidence=1.0)
        with pytest.raises(ValueError, match="n_bootstrap"):
            paired_task_effect([0.1, 0.2], n_bootstrap=0)


# --- decide ---------------------------------------------------------------------


def _effect(delta: float | None, lo: float | None, hi: float | None, tasks: int = 5) -> PairedEffect:
    return PairedEffect(
        tasks=tasks, delta=delta, ci_lower=lo, ci_upper=hi,
        confidence=0.95, n_bootstrap=10, seed=0,
    )


class TestDecide:
    @pytest.mark.parametrize(
        ("delta", "lo", "hi", "verdict"),
        [
            (None, None, None, Verdict.INSUFFICIENT_EVIDENCE),
            (0.4, None, None, Verdict.INSUFFICIENT_EVIDENCE),
            (-0.10, -0.15, -0.05, Verdict.REGRESSION),
            (0.10, 0.05, 0.15, Verdict.IMPROVEMENT),
            (0.01, 0.005, 0.02, Verdict.BELOW_THRESHOLD),
            (-0.02, -0.025, -0.01, Verdict.BELOW_THRESHOLD),
            (0.0, -0.01, 0.02, Verdict.EQUIVALENT),
            (0.02, -0.05, 0.10, Verdict.INCONCLUSIVE),
            (0.0, -0.03, 0.0, Verdict.INCONCLUSIVE),  # touches the threshold
            (0.03, 0.03, 0.03, Verdict.IMPROVEMENT),  # degenerate at the threshold
        ],
    )
    def test_verdict_table(self, delta, lo, hi, verdict):
        assert decide(_effect(delta, lo, hi), threshold=0.03) is verdict

    def test_floating_point_residue_never_decides_significance(self):
        assert not excludes_zero(1e-12, 0.2) and not excludes_zero(-0.2, -1e-12)
        assert excludes_zero(1e-6, 0.2) and excludes_zero(-0.2, -1e-6)
        assert decide(_effect(0.1, 1e-12, 0.2), threshold=0.03) is Verdict.INCONCLUSIVE


# --- compare_runs ---------------------------------------------------------------


class TestCompareRuns:
    def test_hand_derived_paired_delta_and_per_task_rows(self):
        baseline = _pass_batch({"t1": [True, False], "t2": [True, True], "t3": [False, False]})
        candidate = _pass_batch({"t1": [True, True], "t2": [True, False], "t3": [True, False]})
        result = compare_runs(baseline, candidate, n_bootstrap=64, seed=0)
        # per-task means: t1 0.5 -> 1.0, t2 1.0 -> 0.5, t3 0.0 -> 0.5
        assert result.delta == pytest.approx((0.5 - 0.5 + 0.5) / 3)
        # sorted by |delta| descending, then task id
        assert [(r.task_id, r.delta) for r in result.per_task] == [
            ("t1", 0.5), ("t2", -0.5), ("t3", 0.5)
        ]
        assert result.per_task[0].n_baseline == 2 and result.per_task[0].baseline == 0.5
        # every sign assignment of (0.5, 0.5, 0.5) is at least as extreme: p = 1
        assert result.p_value == 1.0 and result.p_value_exact
        assert result.alignment.aligned_by == "content" and result.alignment.compared == 3
        assert result.baseline.trials_gradable == 6 and result.candidate.trials_total == 6
        assert result.method == "paired task bootstrap" and result.unit == "task"

    def test_pairing_cancels_heterogeneous_task_difficulty(self):
        """Ten tasks from 0% to 90% pass rate, every one improved by exactly 10 points."""
        baseline = _pass_batch({f"t{i}": [True] * i + [False] * (10 - i) for i in range(10)})
        candidate = _pass_batch({f"t{i}": [True] * (i + 1) + [False] * (9 - i) for i in range(10)})
        result = compare_runs(baseline, candidate, n_bootstrap=2000, seed=0)
        assert result.delta == pytest.approx(0.1)
        assert result.ci_lower == pytest.approx(0.1) and result.ci_upper == pytest.approx(0.1)
        assert result.verdict is Verdict.IMPROVEMENT and result.exit_code == 0
        # The unpaired trial-level comparison sees the between-task spread as noise.
        a = [1.0 if t.passed else 0.0 for t in baseline.trials]
        b = [1.0 if t.passed else 0.0 for t in candidate.trials]
        _, lo, hi = bootstrap_difference_ci(a, b, n_bootstrap=2000, seed=0)
        assert hi - lo > 0.1

    def test_repeated_trials_of_one_task_are_not_independent_evidence(self):
        baseline = _pass_batch({"only": [False] * 100})
        candidate = _pass_batch({"only": [True] * 100})
        result = compare_runs(baseline, candidate, n_bootstrap=100)
        assert result.delta == 1.0 and result.ci_lower is None
        assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE and result.exit_code == 2
        assert result.significant is None and result.meaningful is True

    def test_grader_selection_changes_the_estimand(self):
        def batch(g1: list[bool], g2: list[bool]) -> TrialBatch:
            b = TrialBatch()
            for i, (a, c) in enumerate(zip(g1, g2, strict=True)):
                b.add_trial(_trial("t", i, passed=a, extra={"g2": (c, 1.0 if c else 0.0)}))
            b.add_trial(_trial("u", 0, passed=True, extra={"g2": (True, 1.0)}))
            b.provenance = _provenance(["t", "u"], graders=("g", "g2"))
            return b

        baseline = batch([True, True, False, False], [True, False, True, False])
        candidate = batch([True, True, True, True], [False, False, False, False])
        both = compare_runs(baseline, candidate, n_bootstrap=64)
        only_g = compare_runs(baseline, candidate, grader="g", n_bootstrap=64)
        only_g2 = compare_runs(baseline, candidate, grader="g2", n_bootstrap=64)
        # trial-level: all graders must pass -> t: 0.25 -> 0.0; grader g: 0.5 -> 1.0; g2: 0.5 -> 0.0
        assert [r.delta for r in both.per_task if r.task_id == "t"] == [-0.25]
        assert [r.delta for r in only_g.per_task if r.task_id == "t"] == [0.5]
        assert [r.delta for r in only_g2.per_task if r.task_id == "t"] == [-0.5]
        assert only_g.grader == "g" and both.grader is None

    def test_lower_is_better_metric_normalises_direction(self):
        def batch(latency: float) -> TrialBatch:
            b = TrialBatch()
            for t in ("a", "b", "c"):
                for i in range(2):
                    b.add_trial(_trial(t, i, passed=True, metrics={"latency_ms": latency + i}))
            b.provenance = _provenance(["a", "b", "c"])
            return b

        result = compare_runs(
            batch(1000.0), batch(900.0), metric="g.latency_ms", direction="lower",
            threshold=50.0, n_bootstrap=64,
        )
        assert result.delta == pytest.approx(100.0) and result.raw_delta == pytest.approx(-100.0)
        assert result.verdict is Verdict.IMPROVEMENT
        assert result.direction is Direction.LOWER and result.grader is None
        slower = compare_runs(
            batch(900.0), batch(1000.0), metric="g.latency_ms", direction="lower",
            threshold=50.0, n_bootstrap=64,
        )
        assert slower.verdict is Verdict.REGRESSION and slower.exit_code == 1
        assert any("set --threshold on the scale" in n for n in compare_runs(
            batch(900.0), batch(1000.0), metric="g.latency_ms", direction="lower", n_bootstrap=64
        ).notes)

    def test_missing_evidence_is_excluded_and_counted_not_zero(self):
        baseline = _pass_batch({"a": [True, True], "b": [True, False], "c": [False, False]})
        candidate = TrialBatch()
        candidate.add_trial(_trial("a", 0, passed=True))
        candidate.add_trial(_trial("a", 1, passed=True, status=TrialStatus.INFRA_ERROR))
        candidate.add_trial(_trial("b", 0, passed=True, metrics={"m": 1.0}))
        candidate.add_trial(_trial("c", 0, status=TrialStatus.INFRA_ERROR))
        candidate.add_trial(_trial("c", 1, status=TrialStatus.SKIPPED))
        candidate.provenance = _provenance(["a", "b", "c"])
        result = compare_runs(baseline, candidate, n_bootstrap=64)
        assert result.candidate.excluded == {
            "infra_error": 2, "grader_error": 0, "not_run": 1, "no_value": 0,
        }
        assert result.candidate.trials_gradable == 2
        assert result.alignment.excluded_no_value_candidate == ["c"]
        assert result.alignment.compared == 2
        assert {r.task_id: r.n_candidate for r in result.per_task} == {"a": 1, "b": 1}

    def test_non_finite_metric_values_are_no_value(self):
        def batch(values: list[float]) -> TrialBatch:
            b = TrialBatch()
            for i, v in enumerate(values):
                b.add_trial(_trial("t", i, passed=True, metrics={"m": v}))
            b.add_trial(_trial("u", 0, passed=True, metrics={"m": 1.0}))
            b.provenance = _provenance(["t", "u"])
            return b

        result = compare_runs(batch([1.0, math.nan, math.inf]), batch([2.0]), metric="g.m", n_bootstrap=64)
        assert result.baseline.excluded["no_value"] == 2
        assert [r for r in result.per_task if r.task_id == "t"][0].n_baseline == 1

    def test_changed_task_content_is_refused_unless_excluded_explicitly(self):
        baseline = _pass_batch({"a": [True], "b": [True], "c": [False]})
        candidate = _pass_batch({"a": [True], "b": [True], "c": [False]}, hashes={"b": "changed"})
        with pytest.raises(ComparisonError, match=r"task sets differ: 1 changed content \(b\)"):
            compare_runs(baseline, candidate)
        result = compare_runs(baseline, candidate, unmatched_tasks="exclude", n_bootstrap=64)
        assert result.alignment.excluded_changed == ["b"] and result.alignment.compared == 2
        assert result.compatibility.status.value == "incompatible"

    def test_added_and_removed_tasks_follow_the_same_policy(self):
        baseline = _pass_batch({"a": [True], "b": [True]})
        candidate = _pass_batch({"a": [True], "c": [True]})
        with pytest.raises(ComparisonError, match="1 only in baseline \\(b\\); 1 only in candidate \\(c\\)"):
            compare_runs(baseline, candidate)
        result = compare_runs(baseline, candidate, unmatched_tasks="exclude", n_bootstrap=64)
        assert result.alignment.excluded_only_baseline == ["b"]
        assert result.alignment.excluded_only_candidate == ["c"]
        assert result.alignment.compared == 1

    def test_different_graders_are_never_compared(self):
        baseline = _pass_batch({"a": [True], "b": [True]}, graders=("g",))
        candidate = _pass_batch({"a": [True], "b": [True]}, graders=("g", "extra"))
        with pytest.raises(ComparisonError, match="graded differently"):
            compare_runs(baseline, candidate, unmatched_tasks="exclude")

    def test_legacy_artifacts_align_by_id_and_say_so(self):
        baseline = _pass_batch({"a": [True, False], "b": [True, True]}, provenance=False)
        candidate = _pass_batch({"a": [True, True], "b": [True, True], "c": [True]})
        with pytest.raises(ComparisonError, match="drop --require-provenance"):
            compare_runs(baseline, candidate, require_provenance=True)
        with pytest.raises(ComparisonError, match="1 only in candidate \\(c\\)"):
            compare_runs(baseline, candidate)
        result = compare_runs(baseline, candidate, unmatched_tasks="exclude", n_bootstrap=64)
        assert result.alignment.aligned_by == "id" and result.alignment.compared == 2
        assert result.compatibility.status.value == "unknown"
        assert any("aligned by id only" in note for note in result.notes)
        assert "What changed: unknown" in "\n".join(result.summary_lines())

    def test_same_inputs_and_seed_reproduce_the_whole_record(self):
        baseline = _pass_batch({f"t{i}": [i % 2 == 0, True, False] for i in range(6)})
        candidate = _pass_batch({f"t{i}": [True, i % 3 == 0, False] for i in range(6)})
        a = compare_runs(baseline, candidate, seed=11, n_bootstrap=300)
        b = compare_runs(baseline, candidate, seed=11, n_bootstrap=300)
        assert a.model_dump() == b.model_dump()
        assert RunComparison.model_validate_json(a.model_dump_json()) == a

    def test_observe_exits_zero_for_evaluated_comparisons_only(self):
        baseline = _pass_batch({"a": [True, False], "b": [False, False]})
        candidate = _pass_batch({"a": [False, True], "b": [True, False]})
        result = compare_runs(baseline, candidate, n_bootstrap=64, observe=True)
        assert result.verdict is Verdict.INCONCLUSIVE and result.exit_code == 0
        empty = compare_runs(
            _pass_batch({"a": [True]}, hashes={"a": "x"}),
            _pass_batch({"a": [True]}, hashes={"a": "y"}),
            unmatched_tasks="exclude", observe=True,
        )
        assert empty.delta is None and empty.exit_code == 2

    def test_rejects_bad_parameters(self):
        baseline = _pass_batch({"a": [True]})
        with pytest.raises(ComparisonError, match="threshold cannot be negative"):
            compare_runs(baseline, baseline, threshold=-1)
        with pytest.raises(ComparisonError, match="unmatched_tasks must be"):
            compare_runs(baseline, baseline, unmatched_tasks="drop")
        with pytest.raises(ComparisonError, match="confidence"):
            compare_runs(baseline, baseline, confidence=2)

    def test_summary_lines_carry_the_json_facts(self):
        baseline = _pass_batch({"a": [True, False], "b": [True, True], "c": [False, False]})
        candidate = _pass_batch({"a": [True, True], "b": [True, True], "c": [True, True]})
        result = compare_runs(baseline, candidate, n_bootstrap=64, baseline_label="v1", candidate_label="v2")
        text = "\n".join(result.summary_lines(top=1))
        assert text.startswith("Compared v2 vs v1 on pass_rate (higher is better): paired task bootstrap over 3 task(s)")
        assert f"delta = {result.delta:+.4f}" in text
        assert f"(B = {result.n_bootstrap}, seed = {result.seed})" in text
        assert f"(exit {result.exit_code})" in text
        assert "What moved (largest first): c +1.000 (n 2/2), and 1 more" in text
        data = json.loads(result.model_dump_json())
        assert data["verdict"] == result.verdict.value and data["exit_code"] == result.exit_code
