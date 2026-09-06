"""Tests for the baseline gate decision (issue #47)."""

import json
from pathlib import Path

import pytest

from tracelens.baselines.comparison import RegressionSeverity
from tracelens.baselines.manager import BaselineManager, TaskBaseline
from tracelens.core.decision_spec import DecisionSpec, InfraConfig
from tracelens.core.outcome import Outcome
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
) -> Trial:
    trial = Trial(task_id=task_id, run_index=run_index, status=status)
    if passed is not None:
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id, grader_id="g", passed=passed,
            score=1.0 if passed else 0.0, grader_error=grader_error,
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
        assert gate.reasons == ["1 task(s) compared; no regression at or above 'moderate'"]

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
