"""Tests for the baseline gate decision (issue #47)."""

import json
from pathlib import Path

import pytest

from tracelens.baselines.comparison import RegressionSeverity
from tracelens.baselines.manager import BaselineManager, TaskBaseline
from tracelens.core.decision_spec import DecisionSpec, InfraConfig
from tracelens.core.outcome import Outcome
from tracelens.core.provenance import (
    CandidateSpec,
    ComponentIdentity,
    MeasurementSetup,
    RunnerSettings,
    RunProvenance,
)
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch, TrialStatus
from tracelens.reporting.gate import (
    EXIT_CODES,
    GateResult,
    GateStatus,
    TaskGateOutcome,
    evaluate_gate,
    per_trial_results,
    spec_from_trials,
)


def _trial(
    task_id: str,
    passed: bool | None = None,
    *,
    status: TrialStatus = TrialStatus.COMPLETED,
    run_index: int = 0,
    grader_error: bool = False,
    spec: DecisionSpec | None = None,
    score: float | None = None,
) -> Trial:
    trial = Trial(task_id=task_id, run_index=run_index, status=status)
    if passed is not None:
        if score is None:
            score = 1.0 if passed else 0.0
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id, grader_id="g", passed=passed,
            score=score, grader_error=grader_error,
        ))
    if spec is not None:
        trial.transcript = Transcript(task_id=task_id, final_output={}, decision_spec=spec)
    return trial


def _runs(task_id: str, passes: list[bool], **kwargs) -> list[Trial]:
    return [_trial(task_id, p, run_index=i, **kwargs) for i, p in enumerate(passes)]


def _batch(*trials: Trial) -> TrialBatch:
    batch = TrialBatch()
    for trial in trials:
        batch.add_trial(trial)
    return batch


def _manager(tmp_path: Path, baselines: dict[str, dict[str, float]]) -> BaselineManager:
    manager = BaselineManager(tmp_path / "baselines.json")
    for task_id, metrics in baselines.items():
        baseline = TaskBaseline(task_id=task_id)
        for name, value in metrics.items():
            baseline.add_metric(name, value, std=0.05, sample_size=10)
        manager.set_baseline(baseline)
    manager.save()
    return manager


class TestEvaluateGate:
    def test_passed(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True, True])),
            _manager(tmp_path, {"t1": {"pass_rate": 1.0}}),
        )
        assert gate.status is GateStatus.PASSED and gate.exit_code == 0
        assert (gate.checked, gate.skipped_no_baseline, gate.blocking_regressions) == (1, 0, 0)
        assert gate.tasks[0].outcome is TaskGateOutcome.CHECKED
        assert gate.tasks[0].compared_trials == 3 and not gate.tasks[0].blocking
        assert gate.summary_line() == (
            "[tracelens] Baseline check: 1 checked, 0 skipped (no baseline), "
            "0 blocking regression(s)"
        )
        assert gate.reasons == [
            "1 task(s) compared; no significant regression at or above 'moderate'"
        ]

    def test_blocked_on_regression(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [False, False, False])),
            _manager(tmp_path, {"t1": {"pass_rate": 1.0}}),
        )
        assert gate.status is GateStatus.BLOCKED and gate.exit_code == 1
        assert gate.blocking_regressions == 1
        task = gate.tasks[0]
        assert task.blocking and task.overall_severity is RegressionSeverity.SEVERE
        assert task.regressions[0].metric_name == "pass_rate"
        assert task.regressions[0].current_mean == 0.0
        assert "1 blocking regression(s) at threshold 'moderate': t1 (severe)" in gate.reasons
        assert gate.summary_line().endswith("1 blocking regression(s)")
        assert "REGRESSION DETECTED [SEVERE]" in task.regression_report().to_ci_output()

    def test_threshold_controls_blocking(self, tmp_path):
        manager = _manager(tmp_path, {"t1": {"pass_rate": 1.0}})
        batch = _batch(*_runs("t1", [False, False, False]))
        lenient = evaluate_gate(batch, manager, threshold=RegressionSeverity.SEVERE)
        assert lenient.status is GateStatus.BLOCKED  # -100% is severe
        assert lenient.threshold is RegressionSeverity.SEVERE

    def test_missing_baseline_is_skipped_unless_required(self, tmp_path):
        manager = _manager(tmp_path, {"t1": {"pass_rate": 1.0}})
        batch = _batch(*_runs("t1", [True, True]), *_runs("t2", [True, True]))
        relaxed = evaluate_gate(batch, manager)
        assert relaxed.status is GateStatus.PASSED
        assert relaxed.skipped_no_baseline == 1
        assert relaxed.tasks_with(TaskGateOutcome.NO_BASELINE)[0].task_id == "t2"

        strict = evaluate_gate(batch, manager, require_baselines=True)
        assert strict.status is GateStatus.BLOCKED and strict.exit_code == 1
        assert strict.reasons == ["--require-baselines set but 1 task(s) have no baseline: t2"]

    def test_unevaluable_when_no_gradable_trials(self, tmp_path):
        batch = _batch(
            _trial("t1", status=TrialStatus.INFRA_ERROR, run_index=0),
            _trial("t1", status=TrialStatus.INFRA_ERROR, run_index=1),
        )
        gate = evaluate_gate(batch, _manager(tmp_path, {"t1": {"pass_rate": 1.0}}))
        assert gate.status is GateStatus.UNEVALUABLE and gate.exit_code == 2
        task = gate.tasks[0]
        assert task.outcome is TaskGateOutcome.NO_GRADABLE_TRIALS
        assert task.excluded_trials == 2
        assert "1 skipped (no gradable trials)" in gate.summary_line()
        assert gate.summary_line().endswith("UNEVALUABLE")
        assert "1 task(s) with no gradable trials: t1" in gate.reasons

    def test_unevaluable_when_no_comparable_metrics(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True])),
            _manager(tmp_path, {"t1": {"domain_quality": 0.9}}),
        )
        assert gate.status is GateStatus.UNEVALUABLE
        task = gate.tasks[0]
        assert task.outcome is TaskGateOutcome.NO_COMPARABLE_METRICS
        assert task.available_metrics == ["mean_score", "pass_rate"]
        assert "no comparable metrics" in gate.summary_line()

    def test_unevaluable_when_nothing_could_be_checked(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True])),
            _manager(tmp_path, {"other": {"pass_rate": 1.0}}),
        )
        assert gate.status is GateStatus.UNEVALUABLE
        assert gate.checked == 0
        assert gate.reasons[0] == "no task could be compared against a baseline"

    def test_unevaluable_takes_precedence_over_blocking(self, tmp_path):
        batch = _batch(
            *_runs("t1", [False, False, False]),
            _trial("t2", status=TrialStatus.INFRA_ERROR),
        )
        gate = evaluate_gate(
            batch, _manager(tmp_path, {"t1": {"pass_rate": 1.0}, "t2": {"pass_rate": 1.0}})
        )
        assert gate.status is GateStatus.UNEVALUABLE and gate.exit_code == 2
        # The observed regression is still recorded, not hidden.
        assert gate.blocking_regressions == 1
        assert gate.tasks[0].blocking

    def test_infra_config_mismatch_is_recorded(self, tmp_path):
        manager = _manager(tmp_path, {"t1": {"pass_rate": 1.0}})
        baseline = manager.get_baseline("t1")
        assert baseline is not None
        baseline.decision_spec = DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=2048))
        manager.set_baseline(baseline)
        current = DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=512))
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True, True], spec=current)), manager
        )
        task = gate.tasks[0]
        assert task.infra_config_mismatch
        assert task.infra_config_diff["memory_hard_limit_mb"] == (2048, 512)

    def test_task_order_follows_task_ids_argument(self, tmp_path):
        manager = _manager(tmp_path, {"a": {"pass_rate": 1.0}, "b": {"pass_rate": 1.0}})
        batch = _batch(*_runs("b", [True]), *_runs("a", [True]))
        gate = evaluate_gate(batch, manager, task_ids=["b", "a"])
        assert [t.task_id for t in gate.tasks] == ["b", "a"]
        assert [t.task_id for t in evaluate_gate(batch, manager).tasks] == ["a", "b"]


class TestGateResultModel:
    def test_not_requested_and_exit_codes(self):
        gate = GateResult.not_requested()
        assert gate.status is GateStatus.NOT_REQUESTED
        assert gate.exit_code == 0 and not gate.requested
        assert EXIT_CODES == {
            GateStatus.NOT_REQUESTED: 0,
            GateStatus.PASSED: 0,
            GateStatus.BLOCKED: 1,
            GateStatus.UNEVALUABLE: 2,
        }

    def test_round_trip_preserves_everything(self, tmp_path):
        manager = _manager(tmp_path, {"t1": {"pass_rate": 1.0}, "t2": {"pass_rate": 1.0}})
        baseline = manager.get_baseline("t1")
        assert baseline is not None
        baseline.decision_spec = DecisionSpec(infra=InfraConfig(cpu_hard_limit=2.0))
        manager.set_baseline(baseline)
        batch = _batch(
            *_runs("t1", [False, False, False], spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=1.0))),
            *_runs("t2", [True, True]),
            *_runs("t3", [True]),
        )
        gate = evaluate_gate(batch, manager, require_baselines=True)
        restored = GateResult.from_dict(json.loads(json.dumps(gate.to_dict())))
        assert restored.to_dict() == gate.to_dict()
        assert restored.status is gate.status and restored.exit_code == gate.exit_code
        assert restored.tasks[0].regressions == gate.tasks[0].regressions
        assert restored.tasks[0].infra_config_diff == {"cpu_hard_limit": (2.0, 1.0)}
        assert restored.summary_line() == gate.summary_line()


class TestHelpers:
    def test_per_trial_results_uses_gradable_trials_only(self):
        trials = [
            _trial("t", True, run_index=0),
            _trial("t", status=TrialStatus.PENDING, run_index=1),
            _trial("t", status=TrialStatus.INFRA_ERROR, run_index=2),
            _trial("t", False, run_index=3, grader_error=True),
            _trial("t", status=TrialStatus.TIMEOUT, run_index=4),
        ]
        results = per_trial_results(trials)
        assert [r["pass_rate"] for r in results] == [1.0, 0.0]  # pass, timeout-as-failure

    def test_spec_from_trials_prefers_latest_and_reports_mix(self):
        old = _trial("t", True, run_index=0, spec=DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=2048)))
        new = _trial("t", True, run_index=1, spec=DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=512)))
        spec, warning = spec_from_trials([old, new])
        assert spec is not None and spec.infra is not None
        assert spec.infra.memory_hard_limit_mb == 512
        assert warning is not None and "mixed decision specs" in warning
        assert spec_from_trials([_trial("t", True)]) == (None, None)
        _, no_warning = spec_from_trials([new, new])
        assert no_warning is None


def test_gate_without_task_argument_uses_all_tasks(tmp_path):
    gate = evaluate_gate(
        _batch(*_runs("t1", [True])), _manager(tmp_path, {"t1": {"pass_rate": 1.0}})
    )
    assert gate.status in (GateStatus.PASSED, GateStatus.BLOCKED)
    assert pytest.approx(gate.noise_band) == 0.03


class TestTaskContentIdentity:
    """Issue #51: a baseline is compared only with the task content it was stored for."""

    @staticmethod
    def _manager_with_hashes(tmp_path: Path, hashes: dict[str, str]) -> BaselineManager:
        manager = BaselineManager(tmp_path / "baselines.json")
        for task_id, task_hash in hashes.items():
            baseline = TaskBaseline(task_id=task_id, task_hash=task_hash)
            baseline.add_metric("pass_rate", 1.0, std=0.05, sample_size=10)
            manager.set_baseline(baseline)
        manager.save()
        return manager

    @staticmethod
    def _provenance_for(task_hashes: dict[str, str]) -> RunProvenance:
        return RunProvenance(
            run_id="r",
            measurement=MeasurementSetup(
                eval_set_hash="e" * 64,
                task_hashes=task_hashes,
                runner=RunnerSettings(
                    num_runs=1, max_concurrency=1, timeout_seconds=1.0, max_infra_retries=0
                ),
            ),
            candidate=CandidateSpec(adapter=ComponentIdentity(class_path="x.A")),
        )

    def test_changed_task_content_makes_the_gate_unevaluable(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True])),
            self._manager_with_hashes(tmp_path, {"t1": "a" * 64}),
            task_hashes={"t1": "b" * 64},
        )
        assert gate.status is GateStatus.UNEVALUABLE and gate.exit_code == 2
        task = gate.tasks[0]
        assert task.outcome is TaskGateOutcome.TASK_CONTENT_CHANGED
        assert task.reason == (
            "task content changed since the baseline was stored "
            "(aaaaaaaaaaaa -> bbbbbbbbbbbb); re-store the baseline for this task"
        )
        assert gate.skipped_task_content_changed == 1 and gate.checked == 0
        assert (
            "1 task(s) whose content changed since their baseline was stored: t1"
            in gate.reasons
        )
        assert "1 skipped (task content changed)" in gate.summary_line()

    def test_matching_content_is_compared_normally(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True])),
            self._manager_with_hashes(tmp_path, {"t1": "a" * 64}),
            task_hashes={"t1": "a" * 64},
        )
        assert gate.status is GateStatus.PASSED and gate.warnings == []

    def test_unhashed_baseline_is_compared_with_a_warning(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True])),
            _manager(tmp_path, {"t1": {"pass_rate": 1.0}}),
            task_hashes={"t1": "a" * 64},
        )
        assert gate.status is GateStatus.PASSED
        assert gate.warnings == [
            "1 baseline(s) carry no task_hash, so a change to their task content cannot "
            "be detected: t1; re-store them from a results file that records provenance"
        ]

    def test_without_current_hashes_nothing_changes(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True])),
            self._manager_with_hashes(tmp_path, {"t1": "a" * 64}),
        )
        assert gate.status is GateStatus.PASSED and gate.warnings == []

    def test_hashes_default_to_the_batch_provenance(self, tmp_path):
        batch = _batch(*_runs("t1", [True, True]))
        batch.provenance = self._provenance_for({"t1": "b" * 64})
        gate = evaluate_gate(batch, self._manager_with_hashes(tmp_path, {"t1": "a" * 64}))
        assert gate.tasks[0].outcome is TaskGateOutcome.TASK_CONTENT_CHANGED

    def test_content_change_makes_the_gate_unevaluable_even_with_a_regression(self, tmp_path):
        manager = self._manager_with_hashes(tmp_path, {"t1": "a" * 64, "t2": "c" * 64})
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True]), *_runs("t2", [False, False])),
            manager,
            task_hashes={"t1": "b" * 64, "t2": "c" * 64},
        )
        assert gate.status is GateStatus.UNEVALUABLE and gate.exit_code == 2
        assert gate.blocking_regressions == 1 and gate.skipped_task_content_changed == 1

    def test_round_trip_keeps_the_new_count(self, tmp_path):
        gate = evaluate_gate(
            _batch(*_runs("t1", [True, True])),
            self._manager_with_hashes(tmp_path, {"t1": "a" * 64}),
            task_hashes={"t1": "b" * 64},
        )
        assert GateResult.from_dict(json.loads(json.dumps(gate.to_dict()))) == gate
        assert GateResult.from_dict({"status": "passed"}).skipped_task_content_changed == 0


def _manager_n(
    tmp_path: Path, baselines: dict[str, float], *, sample_size: int, std: float = 0.0
) -> BaselineManager:
    """Baselines with an explicit pass-rate count: value * sample_size successes."""
    manager = BaselineManager(tmp_path / "baselines.json")
    for task_id, value in baselines.items():
        baseline = TaskBaseline(task_id=task_id)
        baseline.add_metric("pass_rate", value, std=std, sample_size=sample_size)
        manager.set_baseline(baseline)
    manager.save()
    return manager


def _passes(passed: int, total: int) -> list[bool]:
    return [i < passed for i in range(total)]


class TestRunLevelPolicy:
    """Issue #111: one decision per run, held to one significance level."""

    def test_holm_across_tasks_requires_stronger_evidence(self, tmp_path):
        # Two tasks, each 5/5 in the baseline and 2/5 now: p=0.0309 each
        # (boschloo_exact([[5, 2], [0, 3]], "greater")), under alpha on its
        # own but not once both tasks share alpha.
        manager = _manager_n(tmp_path, {"a": 1.0, "b": 1.0}, sample_size=5)
        batch = _batch(*_runs("a", _passes(2, 5)), *_runs("b", _passes(2, 5)))

        holm = evaluate_gate(batch, manager)
        assert holm.status is GateStatus.PASSED and holm.exit_code == 0
        assert holm.multiplicity == "holm" and holm.family_size == 2
        assert holm.alpha == 0.05
        for task in holm.tasks:
            reg = task.regressions[0]
            assert reg.p_value == pytest.approx(0.0309, abs=5e-4)
            assert reg.p_value_adjusted == pytest.approx(0.0618, abs=1e-3)
            assert reg.is_significant is False and reg.underpowered is True
            assert task.blocking is False and task.has_regression is False
        assert holm.blocking_regressions == 0
        assert "2 task(s) show a drop the evidence could not confirm: a, b" in holm.reasons[1]
        assert "2 observed drop(s) not significant" in holm.summary_line()
        assert holm.underpowered_tasks == holm.tasks

        uncorrected = evaluate_gate(batch, manager, multiplicity="none")
        assert uncorrected.status is GateStatus.BLOCKED
        assert uncorrected.blocking_regressions == 2
        assert uncorrected.tasks[0].regressions[0].p_value_adjusted is None
        assert uncorrected.policy_text() == "alpha=0.05 per test, no multiplicity correction"

        # A single test carries no correction at all.
        single = evaluate_gate(_batch(*_runs("a", _passes(2, 5))), manager, task_ids=["a"])
        assert single.status is GateStatus.BLOCKED and single.family_size == 1
        assert single.policy_text() == (
            "alpha=0.05, Holm-adjusted across 1 compared (task, metric) test(s)"
        )

    def test_stronger_per_task_evidence_survives_the_correction(self, tmp_path):
        manager = _manager_n(tmp_path, {"a": 1.0, "b": 1.0}, sample_size=5)
        batch = _batch(*_runs("a", _passes(1, 5)), *_runs("b", _passes(5, 5)))
        gate = evaluate_gate(batch, manager)
        assert gate.status is GateStatus.BLOCKED and gate.blocking_regressions == 1
        reg = gate.tasks[0].regressions[0]
        # boschloo_exact([[5, 1], [0, 4]], alternative="greater")
        assert reg.p_value == pytest.approx(0.0107, abs=5e-4)
        assert reg.p_value_adjusted == pytest.approx(0.0215, abs=5e-4)  # 2 * p, under 0.05
        assert gate.tasks[1].regressions == []

    def test_suite_level_criterion_reports_a_broad_regression_without_blocking(
        self, tmp_path
    ):
        # Twenty tasks each slip from 5/5 to 4/5: no task can show it
        # (p=0.29 each), the suite statistic can (every difference is
        # negative). It is reported, but blocking on it is off by default:
        # the sign-flip p-value assumes the per-task differences are
        # independent, and correlated task outcomes inflate its false-alarm
        # rate well past alpha.
        ids = [f"t{i:02d}" for i in range(20)]
        manager = _manager_n(tmp_path, dict.fromkeys(ids, 1.0), sample_size=5)
        batch = _batch(*[t for task_id in ids for t in _runs(task_id, _passes(4, 5))])

        gate = evaluate_gate(batch, manager)
        assert all(not task.blocking for task in gate.tasks)
        assert gate.status is GateStatus.PASSED and gate.exit_code == 0
        assert gate.blocking_regressions == 0
        suite = {s.metric_name: s for s in gate.suite}
        assert set(suite) == {"pass_rate"}  # the baselines carry pass_rate only
        effect = suite["pass_rate"]
        assert effect.tasks == 20 and effect.delta == pytest.approx(-0.2)
        assert effect.delta_percent == pytest.approx(-20.0)
        assert effect.p_value is not None and effect.p_value < 0.001
        assert effect.ci_upper is not None and effect.ci_upper < 0
        assert effect.severity is RegressionSeverity.SEVERE
        assert effect.is_regression and effect.is_significant
        assert effect.blocking is False and effect.blocking_enabled is False
        assert "reported only" in effect.describe()

    def test_suite_level_blocking_is_available_on_request(self, tmp_path):
        ids = [f"t{i:02d}" for i in range(20)]
        manager = _manager_n(tmp_path, dict.fromkeys(ids, 1.0), sample_size=5)
        batch = _batch(*[t for task_id in ids for t in _runs(task_id, _passes(4, 5))])

        gate = evaluate_gate(batch, manager, suite_blocking=True)
        assert gate.status is GateStatus.BLOCKED and gate.exit_code == 1
        assert gate.blocking_regressions == 1
        effect = next(s for s in gate.suite if s.metric_name == "pass_rate")
        assert effect.blocking and effect.blocking_enabled
        assert gate.reasons[0].startswith("1 suite-level regression(s) at threshold 'moderate': ")
        assert "pass_rate: 1.0000 -> 0.8000 (-20.0%) over 20 task(s)" in gate.reasons[0]
        assert "significant, severe: blocking" in effect.describe()
        # Both criteria live: each is held to half the run's budget.
        assert "split with the suite criterion" in gate.policy_text()

    def test_correlated_tasks_make_the_suite_statistic_overstate_its_evidence(
        self, tmp_path
    ):
        # Why suite-level blocking is off by default. The sign-flip test
        # treats each task's difference as an independent draw. Ten tasks
        # that share one run-level outcome carry one task's worth of
        # evidence, but the statistic reads them as ten, and its p-value
        # falls by three orders of magnitude for no new information.
        ids = [f"t{i:02d}" for i in range(10)]
        manager = _manager_n(tmp_path, dict.fromkeys(ids, 1.0), sample_size=5)
        shared = _passes(4, 5)  # every task lives or dies with the same run
        batch = _batch(*[t for task_id in ids for t in _runs(task_id, shared)])

        gate = evaluate_gate(batch, manager)
        effect = next(s for s in gate.suite if s.metric_name == "pass_rate")
        assert effect.tasks == 10
        assert effect.p_value is not None and effect.p_value < 0.002
        # One task on its own is no evidence at all, and the ten carry the
        # same information: nothing here justifies blocking the run.
        alone = evaluate_gate(
            _batch(*_runs(ids[0], shared)), manager, task_ids=[ids[0]]
        )
        assert alone.suite == []  # one task cannot form the statistic
        assert gate.status is GateStatus.PASSED and effect.blocking is False

    def test_the_suite_criteria_share_one_budget(self, tmp_path):
        # Two stored metrics means two suite criteria. Testing each at the
        # full suite level would spend that half of the budget twice, which
        # is the defect the per-task family was fixed for.
        ids = [f"t{i:02d}" for i in range(20)]
        manager = _manager(tmp_path, {i: {"pass_rate": 1.0, "mean_score": 1.0} for i in ids})
        batch = _batch(*[t for task_id in ids for t in _runs(task_id, _passes(4, 5))])

        gate = evaluate_gate(batch, manager, suite_blocking=True)
        by_metric = {s.metric_name: s for s in gate.suite}
        assert set(by_metric) == {"mean_score", "pass_rate"}
        for effect in by_metric.values():
            assert effect.p_value is not None and effect.p_value_adjusted is not None
            assert effect.p_value_adjusted >= effect.p_value
        # Holm over the two criteria: the smaller raw p-value pays 2x.
        smallest = min(s.p_value for s in gate.suite if s.p_value is not None)
        paid = next(s for s in gate.suite if s.p_value == smallest)
        assert paid.p_value_adjusted == pytest.approx(min(1.0, 2 * smallest))

    def test_the_budget_is_split_only_when_the_suite_can_take_a_share(self, tmp_path):
        # One task means no suite criterion can form, so charging the
        # per-task family half the level would halve the run's sensitivity
        # for a criterion that never runs.
        manager = _manager_n(tmp_path, {"a": 1.0}, sample_size=5)
        alone = evaluate_gate(
            _batch(*_runs("a", _passes(2, 5))), manager, suite_blocking=True
        )
        assert alone.suite == [] and alone.suite_blocking is False
        assert alone.policy_text() == (
            "alpha=0.05, Holm-adjusted across 1 compared (task, metric) test(s)"
        )

        # Two tasks can, so both criteria are live and the budget splits.
        manager2 = _manager_n(tmp_path, {"a": 1.0, "b": 1.0}, sample_size=5)
        both = evaluate_gate(
            _batch(*_runs("a", _passes(2, 5)), *_runs("b", _passes(2, 5))),
            manager2, suite_blocking=True,
        )
        assert both.suite and both.suite_blocking is True
        assert "0.025 for the per-task family" in both.policy_text()
        # The split the run used is recorded, not re-derived on read.
        assert GateResult.from_dict(both.to_dict()).policy_text() == both.policy_text()

    def test_suite_level_criterion_ignores_one_task_among_many(self, tmp_path):
        ids = [f"t{i}" for i in range(10)]
        manager = _manager_n(tmp_path, dict.fromkeys(ids, 1.0), sample_size=5)
        runs = [t for task_id in ids[1:] for t in _runs(task_id, _passes(5, 5))]
        batch = _batch(*_runs(ids[0], _passes(0, 5)), *runs)

        gate = evaluate_gate(batch, manager)
        # The task blocks on its own evidence (p=0.001 <= 0.05/10) ...
        assert gate.tasks[0].blocking and gate.status is GateStatus.BLOCKED
        # ... while the suite criterion does not see one task as a trend.
        effect = next(s for s in gate.suite if s.metric_name == "pass_rate")
        assert effect.is_regression and effect.delta == pytest.approx(-0.1)
        assert effect.p_value == pytest.approx(0.5)
        assert effect.is_significant is False and effect.blocking is False
        assert "not significant" in effect.describe()
        assert gate.blocking_regressions == 1

    def test_suite_level_criterion_reports_no_drop_for_an_unchanged_suite(self, tmp_path):
        ids = ["a", "b", "c"]
        manager = _manager_n(tmp_path, dict.fromkeys(ids, 1.0), sample_size=5)
        batch = _batch(*[t for task_id in ids for t in _runs(task_id, _passes(5, 5))])
        gate = evaluate_gate(batch, manager)
        assert gate.status is GateStatus.PASSED
        assert all(not s.is_regression and "no drop" in s.describe() for s in gate.suite)

    def test_undetectable_sample_sizes_make_the_gate_unevaluable(self, tmp_path):
        # One trial against a baseline that stored one trial: even a total
        # failure reads p=0.25 on the honest counts, so the check could
        # never have blocked whatever the agent did.
        manager = _manager_n(tmp_path, {"a": 1.0, "b": 1.0}, sample_size=1)
        batch = _batch(*_runs("a", [True]), *_runs("b", [True]))

        gate = evaluate_gate(batch, manager)
        assert gate.status is GateStatus.UNEVALUABLE and gate.exit_code == 2
        assert all(task.detectable is False and task.trials_needed == 15 for task in gate.tasks)
        assert gate.reasons == [
            "no checked task has enough trials to detect even a total failure "
            "(0.025 per test over 2 compared (task, metric) test(s) in 2 task(s)), so "
            "the check could not have blocked; "
            "run at least 15 trials per task and store baselines from at least as many"
        ]
        assert gate.summary_line().endswith("UNEVALUABLE")

    def test_enough_tasks_let_the_suite_criterion_decide_at_one_trial(self, tmp_path):
        # Only with suite-level blocking switched on: otherwise the suite
        # statistic cannot rescue a check no task could decide, so the run
        # is unevaluable rather than passing on evidence it never had.
        ids = [f"t{i}" for i in range(6)]
        manager = _manager_n(tmp_path, dict.fromkeys(ids, 1.0), sample_size=1)
        unrescued = evaluate_gate(
            _batch(*[t for task_id in ids for t in _runs(task_id, [True])]), manager
        )
        assert unrescued.status is GateStatus.UNEVALUABLE

        def evaluate(results):
            return evaluate_gate(
                _batch(*[t for task_id in ids for t in _runs(task_id, results)]),
                manager,
                suite_blocking=True,
            )

        passing = evaluate([True])
        assert passing.status is GateStatus.PASSED
        assert all(task.detectable is False for task in passing.tasks)
        failing = evaluate([False])
        assert failing.status is GateStatus.BLOCKED
        assert all(not task.blocking for task in failing.tasks)
        assert next(s for s in failing.suite if s.metric_name == "pass_rate").blocking

    def test_a_thin_baseline_is_named_as_the_limit(self, tmp_path):
        # Both baselines stored one trial, so no number of check trials can
        # decide them. Telling the operator to run more is advice that
        # cannot work; the note has to name the baselines instead.
        manager = BaselineManager(tmp_path / "baselines.json")
        for task_id in ("a", "b"):
            baseline = TaskBaseline(task_id=task_id)
            baseline.add_metric("mean_score", 0.9, std=0.0, sample_size=1)
            manager.set_baseline(baseline)
        manager.save()
        batch = _batch(*[
            _trial(task_id, True, run_index=i, score=0.5)
            for task_id in ("a", "b") for i in range(3)
        ])

        gate = evaluate_gate(batch, manager)
        assert gate.status is GateStatus.UNEVALUABLE
        assert "stored fewer than two trials" in gate.reasons[0]
        assert "more check trials alone cannot decide them" in gate.reasons[0]
        assert "run at least" not in gate.reasons[0]

    def test_partly_undetectable_run_is_a_warning(self, tmp_path):
        manager = BaselineManager(tmp_path / "baselines.json")
        for task_id, n in (("a", 5), ("b", 1)):
            baseline = TaskBaseline(task_id=task_id)
            baseline.add_metric("pass_rate", 1.0, sample_size=n)
            manager.set_baseline(baseline)
        manager.save()
        batch = _batch(*_runs("a", _passes(5, 5)), *_runs("b", [True]))

        gate = evaluate_gate(batch, manager)
        assert gate.status is GateStatus.PASSED
        assert gate.tasks[0].detectable is True and gate.tasks[1].detectable is False
        assert gate.warnings == [
            "1 checked task(s) have too few trials to block on their own at 0.025 per "
            "test: b; run at least 15 trials per task (and store baselines from at "
            "least as many)"
        ]

    def test_bounded_continuous_tests_are_undetectable(self, tmp_path):
        # A score measured as constant over three trials, checked with
        # three identical scores: both sides are constant, so the only test
        # is the exact permutation one and its p-value is fixed at
        # 1/C(6, 3) = 0.05 by the sizes alone. That reaches alpha for a
        # single task but not the 0.025 two tests share under Holm.
        manager = BaselineManager(tmp_path / "baselines.json")
        for task_id in ("a", "b"):
            baseline = TaskBaseline(task_id=task_id)
            baseline.add_metric("mean_score", 0.9, std=0.0, sample_size=3)
            manager.set_baseline(baseline)
        manager.save()
        batch = _batch(*[
            _trial(task_id, True, run_index=i, score=0.9)
            for task_id in ("a", "b") for i in range(3)
        ])

        holm = evaluate_gate(batch, manager)
        assert holm.status is GateStatus.UNEVALUABLE and holm.family_size == 2
        assert all(t.compared_metrics == ["mean_score"] for t in holm.tasks)
        assert all(t.detectable is False and t.trials_needed is None for t in holm.tasks)
        assert holm.reasons[0].startswith(
            "no checked task has enough trials to detect even a total failure "
            "(0.025 per test over 2 compared (task, metric) test(s) in 2 task(s))"
        )
        uncorrected = evaluate_gate(batch, manager, multiplicity="none")
        assert uncorrected.status is GateStatus.PASSED
        assert all(t.detectable is True for t in uncorrected.tasks)

    def test_the_holm_family_spans_every_compared_task_and_metric(self, tmp_path):
        # Task a stores pass_rate and mean_score, task b only pass_rate:
        # three compared (task, metric) pairs, one family, one budget.
        # Giving each metric its own family would hand a suite that stores
        # two near-duplicate metrics two independent chances to block.
        manager = _manager(tmp_path, {
            "a": {"pass_rate": 1.0, "mean_score": 1.0},
            "b": {"pass_rate": 1.0},
        })
        batch = _batch(*_runs("a", _passes(2, 5)), *_runs("b", _passes(5, 5)))

        gate = evaluate_gate(batch, manager)
        assert gate.status is GateStatus.BLOCKED and gate.family_size == 3
        a, b = gate.tasks
        assert a.compared_metrics == ["mean_score", "pass_rate"]
        assert b.compared_metrics == ["pass_rate"] and b.regressions == []
        by_metric = {r.metric_name: r for r in a.regressions}
        # boschloo_exact([[10, 2], [0, 3]], alternative="greater")
        assert by_metric["pass_rate"].p_value == pytest.approx(0.0095, abs=5e-4)
        # Three tests in the family, and the two findings tie, so Holm gives
        # each of them 3 * p rather than the 2 * p a per-metric family gave.
        assert by_metric["pass_rate"].p_value_adjusted == pytest.approx(
            3 * by_metric["pass_rate"].p_value
        )
        assert by_metric["mean_score"].p_value_adjusted == pytest.approx(
            3 * by_metric["mean_score"].p_value
        )
        assert all(r.is_significant for r in a.regressions)

    def test_summary_line_counts_drops_not_tasks(self, tmp_path):
        manager = _manager(tmp_path, {"a": {"pass_rate": 1.0, "mean_score": 1.0}})
        gate = evaluate_gate(_batch(*_runs("a", _passes(4, 5))), manager)
        assert gate.status is GateStatus.PASSED
        assert len(gate.underpowered_tasks) == 1
        assert len(gate.tasks[0].underpowered_regressions) == 2
        assert gate.summary_line().endswith(
            "0 blocking regression(s), 2 observed drop(s) not significant"
        )
    def test_policy_arguments_are_validated(self, tmp_path):
        manager = _manager(tmp_path, {"t1": {"pass_rate": 1.0}})
        batch = _batch(*_runs("t1", [True, True]))
        with pytest.raises(ValueError, match="multiplicity must be one of holm, none"):
            evaluate_gate(batch, manager, multiplicity="bonferroni")
        with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
            evaluate_gate(batch, manager, alpha=1.0)

    def test_round_trip_keeps_the_policy_and_suite_results(self, tmp_path):
        ids = [f"t{i:02d}" for i in range(6)]
        manager = _manager_n(tmp_path, dict.fromkeys(ids, 1.0), sample_size=5)
        batch = _batch(*[t for task_id in ids for t in _runs(task_id, _passes(3, 5))])
        gate = evaluate_gate(batch, manager, multiplicity="none", alpha=0.1, seed=7)
        restored = GateResult.from_dict(json.loads(json.dumps(gate.to_dict())))
        assert restored.to_dict() == gate.to_dict()
        assert (restored.alpha, restored.multiplicity, restored.family_size) == (0.1, "none", 6)
        assert [s.describe() for s in restored.suite] == [s.describe() for s in gate.suite]
        assert restored.suite[0].seed == 7
        assert restored.tasks[0].compared_metrics == ["pass_rate"]
        assert restored.tasks[0].regressions[0].trials_needed == gate.tasks[0].regressions[0].trials_needed

    def test_legacy_gate_json_without_the_policy_still_loads(self):
        legacy = {
            "status": "passed", "exit_code": 0, "threshold": "moderate", "checked": 1,
            "tasks": [{"task_id": "t1", "outcome": "checked", "regressions": [{
                "metric_name": "pass_rate", "baseline_mean": 1.0, "current_mean": 0.0,
                "delta": -1.0, "delta_percent": -100.0, "p_value": None,
                "is_significant": False, "insufficient_data": True, "severity": "severe",
            }]}],
        }
        gate = GateResult.from_dict(legacy)
        assert gate.alpha is None and gate.multiplicity is None and gate.suite == []
        assert gate.policy_text() == "significance policy not recorded"
        assert gate.tasks[0].detectable is None
        assert gate.tasks[0].regressions[0].test is None


class TestGateErrorRates:
    """The contract's error-rate claims, checked against the real gate.

    The exact numbers come from ``scripts/gate_error_rates.py``; here a
    seeded simulation through ``evaluate_gate`` confirms the two claims the
    issue asked for: the null false-alarm rate with flaky tasks stays under
    5 %, and a 1.0 -> 0.4 drop on one task is caught at the stated power.
    """

    @staticmethod
    def _simulate(rng, *, tasks: int, flaky: int, n: int, p_flaky: float, regressed: float | None,
                  tmp_path: Path) -> GateStatus:
        manager = BaselineManager(tmp_path / "baselines.json")
        trials = []
        for index in range(tasks):
            task_id = f"t{index:03d}"
            if index < flaky:
                baseline_passes = int(rng.binomial(n, p_flaky))
                current_rate = p_flaky
            else:
                baseline_passes = n
                current_rate = 1.0
            if index == 0 and regressed is not None:
                baseline_passes, current_rate = n, regressed
            baseline = TaskBaseline(task_id=task_id)
            baseline.add_metric("pass_rate", baseline_passes / n, sample_size=n)
            manager.set_baseline(baseline)
            current_passes = int(rng.binomial(n, current_rate))
            trials.extend(_runs(task_id, _passes(current_passes, n)))
        return evaluate_gate(_batch(*trials), manager).status

    def test_null_false_alarms_stay_under_five_percent(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng(111)
        blocked = sum(
            self._simulate(rng, tasks=50, flaky=10, n=5, p_flaky=0.8, regressed=None,
                           tmp_path=tmp_path / str(i)) is GateStatus.BLOCKED
            for i in range(40)
        )
        assert blocked <= 2  # expected ~0.1% per run under Holm; 5% would be 2 of 40

    def test_one_regressed_task_is_caught_at_the_stated_power(self, tmp_path):
        import numpy as np

        rng = np.random.default_rng(222)
        # Single-task suite at n=5: the contract states 68 % power for 1.0 -> 0.4.
        caught = sum(
            self._simulate(rng, tasks=1, flaky=0, n=5, p_flaky=0.8, regressed=0.4,
                           tmp_path=tmp_path / str(i)) is GateStatus.BLOCKED
            for i in range(60)
        )
        assert 25 <= caught <= 55  # 68 % +/- generous binomial slack over 60 runs
