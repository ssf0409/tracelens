"""A change from a zero baseline mean has no relative size.

``delta_percent`` is the change as a percentage of ``|baseline mean|``, and
the gate's severity is derived from it alone. Against a zero mean the ratio
does not exist: the gate used to report +100 % for any move in either
direction and rate it severe, so a decline read "+100.0%" and blocked at
every ``--fail-on-regression`` level once significant. The contract's
availability rule (``docs/statistical-contract.md``) makes both unavailable
instead, and the decision uses only what is known: a nonzero change reaches
``minor``, and whether it reaches anything above that cannot be said.
"""

import json
from pathlib import Path

import pytest

from tracelens.baselines.comparison import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
    relative_change,
    severity_for,
)
from tracelens.baselines.manager import BaselineManager, TaskBaseline
from tracelens.core.outcome import Outcome
from tracelens.core.trial import Trial, TrialBatch, TrialStatus
from tracelens.reporting import gate as gate_module
from tracelens.reporting.gate import (
    GateStatus,
    SuiteGateResult,
    _suite_results,
    evaluate_gate,
)
from tracelens.reporting.generator import ReportData, ReportGenerator, _gate_rows

DECLINE = [-5.0, -6.0, -4.0, -5.5, -4.5] * 4


def _pnl_baseline() -> TaskBaseline:
    baseline = TaskBaseline(task_id="t")
    baseline.add_metric("pnl", 0.0, std=1.0, sample_size=20)
    return baseline


def _decline_report() -> RegressionReport:
    return RegressionDetector().compare(_pnl_baseline(), [{"pnl": v} for v in DECLINE])


class TestRelativeSize:
    def test_a_ratio_to_zero_is_unavailable(self) -> None:
        assert relative_change(0.3, 0.0) is None
        assert relative_change(-0.3, 0.0) is None
        assert relative_change(0.0, 0.0) is None  # 0/0: undefined, not 0 %
        assert relative_change(-0.2, 0.8) == pytest.approx(-25.0)

    def test_an_unavailable_size_has_no_severity(self) -> None:
        assert severity_for(None) is None
        assert severity_for(-20.0) is RegressionSeverity.SEVERE


class TestDetector:
    def test_a_decline_from_zero_has_no_percentage_and_no_severity(self) -> None:
        report = _decline_report()
        assert report.improvements == []
        finding = report.regressions[0]
        assert finding.delta == pytest.approx(-5.0)
        assert finding.delta_percent is None
        assert finding.severity is None
        assert finding.is_significant

    def test_an_improvement_from_zero_is_not_rated_severe(self) -> None:
        report = RegressionDetector().compare(
            _pnl_baseline(), [{"pnl": v} for v in [0.05, 0.04, 0.06, 0.05] * 5]
        )
        assert report.regressions == []
        finding = report.improvements[0]
        assert finding.delta == pytest.approx(0.05)
        assert finding.delta_percent is None
        assert finding.severity is None

    def test_every_move_from_zero_is_reported_because_no_floor_can_apply(self) -> None:
        detector = RegressionDetector(min_delta_percent=50.0)
        report = detector.compare(_pnl_baseline(), [{"pnl": -0.01}] * 5)
        assert [r.metric_name for r in report.regressions] == ["pnl"]

    def test_no_move_from_zero_is_not_a_finding(self) -> None:
        report = RegressionDetector().compare(_pnl_baseline(), [{"pnl": 0.0}] * 5)
        assert report.regressions == [] and report.improvements == []

    def test_a_finding_without_a_size_round_trips_through_json(self) -> None:
        finding = _decline_report().regressions[0]
        data = json.loads(json.dumps(finding.model_dump(mode="json"), allow_nan=False))
        assert data["delta_percent"] is None and data["severity"] is None
        restored = MetricRegression.model_validate(data)
        assert restored.delta_percent is None and restored.severity is None


class TestReportDecision:
    def test_the_severity_cannot_be_placed_above_minor(self) -> None:
        report = _decline_report()
        assert report.has_regression
        assert report.unsized_regressions == report.regressions
        # The worst *known* severity: the unsized finding contributes none.
        assert report.overall_severity is RegressionSeverity.NONE
        assert report.block_verdict(RegressionSeverity.MINOR) is True
        assert report.block_verdict(RegressionSeverity.MODERATE) is None
        assert report.block_verdict(RegressionSeverity.SEVERE) is None

    def test_should_block_ci_fails_closed_on_an_unknown_severity(self) -> None:
        # A boolean has no third answer, and a finding that may reach the
        # threshold must not read as a pass.
        report = _decline_report()
        assert report.should_block_ci() is True
        assert report.should_block_ci(RegressionSeverity.SEVERE) is True

    def test_a_known_block_decides_regardless_of_an_unsized_finding(self) -> None:
        baseline = _pnl_baseline()
        baseline.add_metric("pass_rate", 1.0, std=0.0, sample_size=20)
        report = RegressionDetector().compare(
            baseline, [{"pnl": v, "pass_rate": 0.0} for v in DECLINE]
        )
        assert report.overall_severity is RegressionSeverity.SEVERE
        assert report.block_verdict(RegressionSeverity.SEVERE) is True

    def test_an_unsized_finding_that_is_not_significant_does_not_block(self) -> None:
        report = RegressionDetector().compare(_pnl_baseline(), [{"pnl": -0.1}, {"pnl": 0.1}, {"pnl": -0.2}])
        assert report.regressions and not report.regressions[0].is_significant
        assert report.unsized_regressions == []
        assert report.block_verdict(RegressionSeverity.MODERATE) is False
        assert report.should_block_ci() is False

    def test_text_says_unavailable_instead_of_inventing_a_percentage(self) -> None:
        report = _decline_report()
        for text in (report.to_ci_output(), report.summary):
            assert "+100.0%" not in text
            assert "N/A relative to 0" in text
        assert "REGRESSION DETECTED [SEVERITY N/A]" in report.to_ci_output()
        assert "[severity N/A]" in report.summary


def _trial(task_id: str, run_index: int, *, passed: bool, score: float) -> Trial:
    trial = Trial(task_id=task_id, run_index=run_index, status=TrialStatus.COMPLETED)
    trial.add_outcome(Outcome(trial_id=trial.trial_id, grader_id="g", passed=passed, score=score))
    return trial


def _batch(runs: dict[str, list[tuple[bool, float]]]) -> TrialBatch:
    batch = TrialBatch()
    for task_id, outcomes in runs.items():
        for i, (passed, score) in enumerate(outcomes):
            batch.add_trial(_trial(task_id, i, passed=passed, score=score))
    return batch


def _cost_manager(tmp_path: Path, *, with_collapse: bool = False) -> BaselineManager:
    """``cost`` holds a lower-is-better ``mean_score`` that stood at 0."""
    manager = BaselineManager(tmp_path / "baselines.json")
    cost = TaskBaseline(task_id="cost")
    cost.add_metric(
        "mean_score", 0.0, std=0.1, sample_size=20, higher_is_better=False, is_rate=False
    )
    manager.set_baseline(cost)
    if with_collapse:
        collapse = TaskBaseline(task_id="collapse")
        collapse.add_metric("pass_rate", 1.0, std=0.0, sample_size=20)
        manager.set_baseline(collapse)
    manager.save()
    return manager


COST_RISE = [(True, s) for s in [0.5, 0.6, 0.4, 0.55, 0.45]]


class TestGate:
    def test_an_unsized_regression_makes_an_otherwise_passing_gate_unevaluable(
        self, tmp_path: Path
    ) -> None:
        gate = evaluate_gate(_batch({"cost": COST_RISE}), _cost_manager(tmp_path))
        task = gate.tasks[0]
        finding = next(r for r in task.regressions if r.metric_name == "mean_score")
        assert finding.is_significant and finding.delta_percent is None
        assert finding.severity is None
        assert task.blocking is False
        assert gate.status is GateStatus.UNEVALUABLE and gate.exit_code == 2
        assert gate.blocking_regressions == 0
        reason = gate.reasons[-1]
        assert "cost (mean_score)" in reason
        assert "zero baseline" in reason and "'moderate'" in reason

    def test_at_minor_any_significant_change_from_zero_blocks(self, tmp_path: Path) -> None:
        gate = evaluate_gate(
            _batch({"cost": COST_RISE}),
            _cost_manager(tmp_path),
            threshold=RegressionSeverity.MINOR,
        )
        assert gate.status is GateStatus.BLOCKED and gate.exit_code == 1
        assert gate.tasks[0].blocking
        # The blocking task is named with an unknown severity, not "none".
        reason = next(r for r in gate.reasons if "blocking regression(s)" in r)
        assert "cost (severity N/A)" in reason

    def test_a_known_block_elsewhere_decides_the_run(self, tmp_path: Path) -> None:
        gate = evaluate_gate(
            _batch({"cost": COST_RISE, "collapse": [(False, 0.0)] * 5}),
            _cost_manager(tmp_path, with_collapse=True),
        )
        assert gate.status is GateStatus.BLOCKED and gate.exit_code == 1
        assert [t.task_id for t in gate.tasks if t.blocking] == ["collapse"]

    def test_the_decision_survives_a_round_trip(self, tmp_path: Path) -> None:
        gate = evaluate_gate(_batch({"cost": COST_RISE}), _cost_manager(tmp_path))
        restored = type(gate).from_dict(json.loads(json.dumps(gate.to_dict(), allow_nan=False)))
        assert restored.status is GateStatus.UNEVALUABLE
        finding = next(r for r in restored.tasks[0].regressions if r.metric_name == "mean_score")
        assert finding.delta_percent is None and finding.severity is None

    def test_reports_render_unavailable_not_plus_100(self, tmp_path: Path) -> None:
        batch = _batch({"cost": COST_RISE})
        gen = ReportGenerator(k_values=[1], consistency_k_values=[2])
        report = gen.build_report(batch)
        report.gate = evaluate_gate(batch, _cost_manager(tmp_path))
        rows = {row[1]: row for row in _gate_rows(report.gate)}
        assert rows["mean_score"][4] == "N/A relative to 0"
        assert rows["mean_score"][5] == "N/A"
        assert "severity N/A" in rows["mean_score"][7]
        ci = gen.render_ci_summary(report)
        for text in (gen.render_markdown(report), gen.render_html(report), ci):
            assert "+100.0%" not in text
            assert "N/A relative to 0" in text
        assert "significant drop, severity N/A (zero baseline mean): cost mean_score" in ci
        assert "observed drop, not blocking: cost mean_score" not in ci

    def test_legacy_reports_render_an_unsized_finding(self) -> None:
        data = ReportData(regression_report=_decline_report())
        html = ReportGenerator().render_html(data)
        assert "+100.0%" not in html
        assert "<td>N/A relative to 0</td>" in html and "<td>N/A</td>" in html
        assert ">SEVERITY N/A</span>" in html
        assert "REGRESSION [SEVERITY N/A]" in ReportGenerator().render_ci_summary(data)


def _suite(pairs: list[tuple[float, float]], threshold: RegressionSeverity) -> SuiteGateResult:
    (result,) = _suite_results(
        {"pnl": pairs},
        alpha=0.05,
        min_delta_percent=5.0,
        threshold=threshold,
        noise_band=0.03,
        any_infra_mismatch=False,
        seed=0,
        blocking_enabled=True,
    )
    return result


# Eight tasks whose baseline means average to exactly 0, each lower now.
SUITE_DECLINE = [(b, b - 1.0 - 0.1 * i) for i, b in enumerate([0.5, -0.5] * 4)]


class TestSuiteCriterion:
    def test_a_suite_decline_from_a_zero_mean_has_no_size(self) -> None:
        result = _suite(SUITE_DECLINE, RegressionSeverity.MODERATE)
        assert result.baseline_mean == 0.0 and result.delta < 0
        assert result.delta_percent is None and result.severity is None
        assert result.is_regression and result.is_significant
        assert result.blocking is False
        text = result.describe()
        assert "N/A relative to 0" in text and "+100.0%" not in text
        assert "severity N/A" in text

    def test_at_minor_the_suite_criterion_blocks(self) -> None:
        assert _suite(SUITE_DECLINE, RegressionSeverity.MINOR).blocking is True

    def test_the_suite_result_round_trips(self) -> None:
        result = _suite(SUITE_DECLINE, RegressionSeverity.MODERATE)
        data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert data["delta_percent"] is None and data["severity"] is None
        restored = SuiteGateResult.from_dict(data)
        assert restored.delta_percent is None and restored.severity is None

    def test_a_live_unsized_suite_regression_keeps_the_gate_from_passing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Trial scores are bounded to [0, 1], so a run cannot produce this
        # case through the CLI metrics; the decision rule must hold anyway.
        unsized = _suite(SUITE_DECLINE, RegressionSeverity.MODERATE)
        monkeypatch.setattr(gate_module, "_suite_results", lambda *_a, **_k: [unsized])
        manager = BaselineManager(tmp_path / "baselines.json")
        steady = TaskBaseline(task_id="steady")
        steady.add_metric("pass_rate", 1.0, std=0.0, sample_size=20)
        manager.set_baseline(steady)
        gate = evaluate_gate(
            _batch({"steady": [(True, 1.0)] * 5}), manager, suite_blocking=True
        )
        assert gate.status is GateStatus.UNEVALUABLE
        assert "suite-level pnl" in gate.reasons[-1]
