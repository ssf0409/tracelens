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
    VERDICT_EXIT_CODES,
    ComparisonError,
    Direction,
    MetricSelector,
    PairedEffect,
    RunComparison,
    Verdict,
    alpha_for,
    can_reach_level,
    compare_runs,
    decide,
    excludes_zero,
    is_significant,
    min_attainable_p,
    min_tasks_for,
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


def _effect(
    delta: float | None,
    lo: float | None,
    hi: float | None,
    p: float | None = 0.001,
    *,
    tasks: int = 20,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
) -> PairedEffect:
    return PairedEffect(
        tasks=tasks, delta=delta, ci_lower=lo, ci_upper=hi,
        p_value=None if lo is None else p, confidence=confidence,
        n_bootstrap=n_bootstrap, seed=0,
    )


def _exit(effect: PairedEffect, threshold: float = 0.03) -> int:
    return VERDICT_EXIT_CODES[decide(effect, threshold)]


class TestDecide:
    @pytest.mark.parametrize(
        ("delta", "lo", "hi", "p", "verdict"),
        [
            (None, None, None, None, Verdict.INSUFFICIENT_EVIDENCE),
            (0.4, None, None, None, Verdict.INSUFFICIENT_EVIDENCE),
            (-0.10, -0.15, -0.05, 0.001, Verdict.REGRESSION),
            (0.10, 0.05, 0.15, 0.001, Verdict.IMPROVEMENT),
            (0.01, 0.005, 0.02, 0.001, Verdict.BELOW_THRESHOLD),
            (-0.02, -0.025, -0.01, 0.001, Verdict.BELOW_THRESHOLD),  # harmful bound inside
            (0.0, -0.01, 0.02, 0.6, Verdict.EQUIVALENT),
            (0.02, -0.05, 0.10, 0.4, Verdict.INCONCLUSIVE),
            (0.0, -0.03, 0.0, 0.5, Verdict.INCONCLUSIVE),  # touches the threshold
            (0.03, 0.03, 0.03, 0.001, Verdict.IMPROVEMENT),  # degenerate at the threshold
        ],
    )
    def test_verdict_table(self, delta, lo, hi, p, verdict):
        assert decide(_effect(delta, lo, hi, p), threshold=0.03) is verdict

    @pytest.mark.parametrize(
        ("delta", "lo", "hi", "p", "verdict"),
        [
            # The interval excludes 0, but the sign-flip test does not agree.
            (-0.10, -0.15, -0.05, 0.06, Verdict.INCONCLUSIVE),
            (0.10, 0.05, 0.15, 0.06, Verdict.INCONCLUSIVE),
            (0.01, 0.005, 0.02, 0.06, Verdict.EQUIVALENT),
            # The p-value is small, but the interval includes 0.
            (-0.10, -0.20, 0.01, 0.001, Verdict.INCONCLUSIVE),
            # A p-value at the level agrees with it.
            (-0.10, -0.15, -0.05, 0.05, Verdict.REGRESSION),
        ],
    )
    def test_significance_needs_the_interval_and_the_p_value_to_agree(
        self, delta, lo, hi, p, verdict
    ):
        effect = _effect(delta, lo, hi, p)
        assert decide(effect, threshold=0.03) is verdict
        assert is_significant(effect) is (verdict in (Verdict.REGRESSION,))

    @pytest.mark.parametrize(
        ("before", "after"),
        [
            # The issue's pairs, and two more like them: the harmful bound stays at
            # or past -tau while the interval moves off zero toward harm. The old
            # table passed every "after" (significant, |delta| < tau); none may.
            ((-0.024, -0.050, 0.001), (-0.025, -0.050, -0.001)),
            ((-0.024, -0.200, 0.001), (-0.029, -0.200, -0.001)),
            ((-0.010, -0.030, 0.010), (-0.020, -0.030, -0.005)),
            ((-0.020, -0.100, 0.020), (-0.029, -0.100, -0.001)),
        ],
    )
    def test_more_certainty_of_harm_never_turns_exit_2_into_exit_0(self, before, after):
        assert decide(_effect(*before, p=0.4), 0.03) is Verdict.INCONCLUSIVE
        assert decide(_effect(*after), 0.03) is Verdict.INCONCLUSIVE
        assert _exit(_effect(*before, p=0.4)) == _exit(_effect(*after)) == 2

    def test_every_passing_verdict_rules_out_a_regression_of_the_threshold(self):
        grid = [x / 100 for x in range(-12, 13)]
        for lo, hi in product(grid, grid):
            if lo > hi:
                continue
            for delta, p in product((lo, (lo + hi) / 2, hi), (0.001, 0.05, 0.2)):
                effect = _effect(delta, lo, hi, p)
                if _exit(effect) == 0:
                    assert lo > -0.03, (delta, lo, hi, p, decide(effect, 0.03))
                if _exit(effect) == 1:
                    worse = _effect(delta - 0.02, lo - 0.02, hi - 0.02, p)
                    assert _exit(worse) == 1, (delta, lo, hi, p)

    def test_floating_point_residue_never_decides_significance(self):
        assert not excludes_zero(1e-12, 0.2) and not excludes_zero(-0.2, -1e-12)
        assert excludes_zero(1e-6, 0.2) and excludes_zero(-0.2, -1e-6)
        assert decide(_effect(0.1, 1e-12, 0.2), threshold=0.03) is Verdict.INCONCLUSIVE


class TestEvidenceFloor:
    """No verdict below the tasks at which the exact test can reach the level (#112)."""

    @pytest.mark.parametrize(("confidence", "tasks"), [(0.95, 6), (0.90, 5), (0.99, 8), (0.5, 2)])
    def test_min_tasks_is_where_the_exact_test_first_reaches_the_level(self, confidence, tasks):
        assert min_tasks_for(confidence) == tasks
        assert 2.0 ** (1 - tasks) <= alpha_for(confidence)
        assert tasks == 2 or 2.0 ** (2 - tasks) > alpha_for(confidence)

    def test_min_tasks_rejects_a_confidence_outside_0_and_1(self):
        for confidence in (0.0, 1.0, 1 - 1e-13):
            with pytest.raises(ValueError, match="confidence"):
                min_tasks_for(confidence)

    def test_min_attainable_p(self):
        assert min_attainable_p(0, 10_000) is None
        assert min_attainable_p(1, 10_000) == 1.0
        assert min_attainable_p(2, 10_000) == 0.5
        assert min_attainable_p(6, 10_000) == 1 / 32
        # 13 tasks are sampled (2^13 > 10000); the exact floor is still higher.
        assert min_attainable_p(13, 10_000) == 2.0**-12
        # 20 tasks: a sampled estimate never goes below 1 / (B + 1).
        assert min_attainable_p(20, 10_000) == 1 / 10_001
        assert min_attainable_p(5, 20) == 1 / 16
        assert min_attainable_p(20, 10) == 1 / 11

    @pytest.mark.parametrize("tasks", [2, 3, 5, 6, 8])
    def test_the_floor_is_what_the_exact_test_returns_on_the_strongest_data(self, tasks):
        effect = paired_task_effect([-0.5] * tasks, n_bootstrap=1000, seed=0)
        assert effect.p_value_exact and effect.p_value == min_attainable_p(tasks, 1000)

    @pytest.mark.parametrize("tasks", [2, 3, 4, 5])
    def test_no_verdict_below_the_floor_however_decisive_the_data(self, tasks):
        # Every task went from always passing to always failing.
        effect = paired_task_effect([-1.0] * tasks, n_bootstrap=10_000, seed=0)
        assert effect.ci_lower == effect.ci_upper == -1.0
        assert effect.p_value == 2 / 2**tasks
        assert is_significant(effect) is False and not can_reach_level(effect)
        assert decide(effect, 0.03) is Verdict.INSUFFICIENT_EVIDENCE

    def test_six_tasks_are_enough_at_95_percent_and_eight_at_99(self):
        six = paired_task_effect([-1.0] * 6, n_bootstrap=10_000, seed=0)
        assert six.p_value == 1 / 32 and decide(six, 0.03) is Verdict.REGRESSION
        strict = paired_task_effect([-1.0] * 6, confidence=0.99, n_bootstrap=10_000, seed=0)
        assert decide(strict, 0.03) is Verdict.INSUFFICIENT_EVIDENCE
        eight = paired_task_effect([-1.0] * 8, confidence=0.99, n_bootstrap=10_000, seed=0)
        assert decide(eight, 0.03) is Verdict.REGRESSION

    def test_too_few_sampled_sign_flips_cannot_resolve_the_level(self):
        effect = paired_task_effect([-1.0] * 20, n_bootstrap=10, seed=0)
        assert not effect.p_value_exact and effect.p_value is not None
        assert effect.p_value >= 1 / 11
        assert decide(effect, 0.03) is Verdict.INSUFFICIENT_EVIDENCE


class TestCalibration:
    """Under no change the verdict calls a regression no more often than the level.

    The paired differences are drawn around zero with sd 0.1 (the issue's
    null simulation); the interval alone called a regression in 21 % of such
    runs at two tasks and 6.7 % at six (``scripts/compare_error_rates.py``).
    """

    @pytest.mark.parametrize("tasks", [6, 10, 20])
    def test_null_false_verdict_rates_stay_at_or_below_the_level(self, tasks):
        rng = np.random.default_rng(112_000 + tasks)
        runs = 1000
        regressions = improvements = interval_alone = 0
        for i in range(runs):
            diffs = rng.normal(0.0, 0.1, size=tasks)
            effect = paired_task_effect(diffs.tolist(), n_bootstrap=1000, seed=i)
            verdict = decide(effect, 0.03)
            regressions += verdict is Verdict.REGRESSION
            improvements += verdict is Verdict.IMPROVEMENT
            assert effect.delta is not None and effect.ci_lower is not None
            assert effect.ci_upper is not None
            interval_alone += excludes_zero(effect.ci_lower, effect.ci_upper) and (
                effect.delta <= -0.03
            )
        assert regressions / runs <= 0.05
        assert improvements / runs <= 0.05
        if tasks == 6:
            # Control: on the same runs the interval alone (the old table)
            # calls a regression more often than the level allows.
            assert interval_alone / runs > 0.05

    @pytest.mark.parametrize("tasks", [2, 3, 4, 5])
    def test_no_null_run_below_the_floor_gets_a_verdict(self, tasks):
        rng = np.random.default_rng(112_000 + tasks)
        for i in range(200):
            diffs = rng.normal(0.0, 0.1, size=tasks)
            effect = paired_task_effect(diffs.tolist(), n_bootstrap=1000, seed=i)
            assert decide(effect, 0.03) is Verdict.INSUFFICIENT_EVIDENCE


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
        tasks = ["a", "b", "c", "d", "e", "f"]

        def batch(latency: float) -> TrialBatch:
            b = TrialBatch()
            for t in tasks:
                for i in range(2):
                    b.add_trial(_trial(t, i, passed=True, metrics={"latency_ms": latency + i}))
            b.provenance = _provenance(tasks)
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
        baseline = _pass_batch({f"t{i}": [True, False] for i in range(6)})
        candidate = _pass_batch({
            "t0": [True, True], "t1": [False, False], "t2": [True, True],
            "t3": [False, False], "t4": [True, False], "t5": [False, True],
        })
        result = compare_runs(baseline, candidate, n_bootstrap=64, observe=True)
        assert result.verdict is Verdict.INCONCLUSIVE and result.exit_code == 0
        few = compare_runs(
            _pass_batch({"a": [True], "b": [True]}), _pass_batch({"a": [False], "b": [False]}),
            n_bootstrap=64, observe=True,
        )
        assert few.verdict is Verdict.INSUFFICIENT_EVIDENCE and few.exit_code == 0
        empty = compare_runs(
            _pass_batch({"a": [True]}, hashes={"a": "x"}),
            _pass_batch({"a": [True]}, hashes={"a": "y"}),
            unmatched_tasks="exclude", observe=True,
        )
        assert empty.delta is None and empty.exit_code == 2

    def test_two_tasks_give_no_verdict_and_say_what_one_would_take(self):
        """The issue's CLI case: all-pass against all-fail on two tasks."""
        baseline = _pass_batch({"a": [True] * 5, "b": [True] * 5})
        candidate = _pass_batch({"a": [False] * 5, "b": [False] * 5})
        result = compare_runs(baseline, candidate, n_bootstrap=10_000)
        assert result.delta == result.ci_lower == result.ci_upper == -1.0
        assert result.p_value == 0.5 and result.p_value_exact
        assert result.verdict is Verdict.INSUFFICIENT_EVIDENCE and result.exit_code == 2
        assert result.interval_excludes_zero is True and result.significant is False
        assert result.min_attainable_p == 0.5 and result.min_tasks == 6
        text = "\n".join(result.summary_lines())
        assert (
            "readings: not significant (the interval excludes 0, but too few tasks for p), "
            "|delta| >= threshold 0.03, interval beyond -0.03"
        ) in text
        assert (
            "evidence: with 2 paired task(s) the sign-flip test cannot give p below 0.5000; "
            "a 95% verdict needs p <= 0.05, which takes at least 6 tasks"
        ) in text
        assert "Verdict: insufficient evidence (exit 2)" in text
        data = json.loads(result.model_dump_json())
        assert data["min_attainable_p"] == 0.5 and data["min_tasks"] == 6
        assert data["significant"] is False and data["interval_excludes_zero"] is True

    def test_the_same_collapse_on_six_tasks_is_a_regression(self):
        baseline = _pass_batch({f"t{i}": [True] * 5 for i in range(6)})
        candidate = _pass_batch({f"t{i}": [False] * 5 for i in range(6)})
        result = compare_runs(baseline, candidate, n_bootstrap=10_000)
        assert result.p_value == 1 / 32 and result.p_value_exact
        assert result.significant is True and result.min_attainable_p == 1 / 32
        assert result.verdict is Verdict.REGRESSION and result.exit_code == 1
        assert "evidence:" not in "\n".join(result.summary_lines())
        # One task that did not move carries no sign, so five tasks' worth of
        # evidence cannot reach 5 %: p = 4 / 64.
        held = compare_runs(
            baseline,
            _pass_batch({**{f"t{i}": [False] * 5 for i in range(5)}, "t5": [True] * 5}),
            n_bootstrap=10_000,
        )
        assert held.p_value == 1 / 16 and held.interval_excludes_zero and not held.significant
        assert held.verdict is Verdict.INCONCLUSIVE and held.exit_code == 2
        assert "readings: not significant (the interval excludes 0, but p > 0.05)" in (
            "\n".join(held.summary_lines())
        )

    def test_readings_name_whichever_side_disagrees(self):
        baseline, candidate = self._metric_batches([0.01] * 6 + [0.02] * 6)
        result = compare_runs(baseline, candidate, metric="g.m", n_bootstrap=10_000)
        # The rendering reads the recorded fields; pose the rarer disagreement.
        posed = result.model_copy(update={
            "ci_lower": -0.01, "ci_upper": 0.2, "interval_excludes_zero": False,
            "significant": False, "p_value": 0.01, "verdict": Verdict.INCONCLUSIVE,
        })
        assert (
            "readings: not significant (p <= 0.05, but the interval includes 0), "
            "|delta| < threshold 0.03, interval reaches +0.03"
        ) in "\n".join(posed.summary_lines())

    def test_the_shortfall_names_whichever_is_short_tasks_or_draws(self):
        six_down = (
            _pass_batch({f"t{i}": [True] * 3 for i in range(6)}),
            _pass_batch({f"t{i}": [False] * 3 for i in range(6)}),
        )
        # Six tasks are enough, but ten sampled sign flips cannot resolve 5 %.
        draws = compare_runs(*six_down, n_bootstrap=10)
        assert draws.verdict is Verdict.INSUFFICIENT_EVIDENCE and draws.exit_code == 2
        assert draws.min_attainable_p == pytest.approx(1 / 11) and draws.min_tasks == 6
        assert (
            "evidence: with B = 10 sampled sign flips the test cannot give p below 0.0909; "
            "a 95% verdict needs p <= 0.05, which takes B >= 19"
        ) in "\n".join(draws.summary_lines())
        # One task: no interval, and the evidence line says why no verdict.
        one = compare_runs(_pass_batch({"a": [True]}), _pass_batch({"a": [False]}))
        text = "\n".join(one.summary_lines())
        assert "delta = -1.0000; no interval (fewer than 2 tasks)" in text
        assert "with 1 paired task(s) the sign-flip test cannot give p below 1.0000" in text
        # No task with a value on both sides: nothing to say about the test.
        none = compare_runs(
            _pass_batch({"a": [True]}, hashes={"a": "x"}),
            _pass_batch({"a": [True]}, hashes={"a": "y"}),
            unmatched_tasks="exclude",
        )
        text = "\n".join(none.summary_lines())
        assert "delta: n/a (no task has a value on both sides)" in text
        assert none.min_attainable_p is None and "evidence:" not in text

    @staticmethod
    def _metric_batches(drops: list[float]) -> tuple[TrialBatch, TrialBatch]:
        tasks = [f"t{i:02d}" for i in range(len(drops))]
        baseline, candidate = TrialBatch(), TrialBatch()
        for task_id, drop in zip(tasks, drops, strict=True):
            baseline.add_trial(_trial(task_id, 0, passed=True, metrics={"m": 1.0}))
            candidate.add_trial(_trial(task_id, 0, passed=True, metrics={"m": 1.0 - drop}))
        baseline.provenance = _provenance(tasks)
        candidate.provenance = _provenance(tasks)
        return baseline, candidate

    def test_a_small_significant_drop_that_may_be_larger_is_inconclusive(self):
        """The non-monotone row: significant, |delta| < tau, but -tau is inside the interval."""
        baseline, candidate = self._metric_batches([0.005] * 6 + [0.05] * 6)
        result = compare_runs(baseline, candidate, metric="g.m", threshold=0.03, n_bootstrap=10_000)
        assert result.delta == pytest.approx(-0.0275)
        assert result.significant is True and result.ci_lower <= -0.03
        assert result.verdict is Verdict.INCONCLUSIVE and result.exit_code == 2
        assert "readings: significant, |delta| < threshold 0.03, interval reaches -0.03" in (
            "\n".join(result.summary_lines())
        )
        # The same drop, pinned inside the threshold, is below it and passes.
        baseline, candidate = self._metric_batches([0.01] * 6 + [0.02] * 6)
        tight = compare_runs(baseline, candidate, metric="g.m", threshold=0.03, n_bootstrap=10_000)
        assert tight.significant is True and tight.ci_lower > -0.03
        assert tight.verdict is Verdict.BELOW_THRESHOLD and tight.exit_code == 0
        assert "interval inside (-0.03, +0.03)" in "\n".join(tight.summary_lines())

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
