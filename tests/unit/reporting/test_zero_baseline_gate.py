"""A change from a zero baseline mean has no percentage, but it has a size.

``delta_percent`` is the change as a percentage of ``|baseline mean|``. Against
a zero mean that number does not exist, and the gate used to print +100 % for
any move in either direction, so a decline read "+100.0%". The number is now
unavailable (``None``, printed ``N/A relative to 0``). The severity is not:
it bins the size of the relative change, and a nonzero change from 0 is
unbounded relative to it, so it is beyond every bin and ``severe``. A
significant one blocks at every ``--fail-on-regression`` level, as it did on
main, and a baseline of 1e-300 gives the same verdict.
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

    def test_an_unbounded_change_is_beyond_every_bin(self) -> None:
        assert severity_for(None) is RegressionSeverity.SEVERE
        assert severity_for(-20.0) is RegressionSeverity.SEVERE


class TestDetector:
    def test_a_decline_from_zero_has_no_percentage_and_is_severe(self) -> None:
        report = _decline_report()
        assert report.improvements == []
        finding = report.regressions[0]
        assert finding.delta == pytest.approx(-5.0)
        assert finding.delta_percent is None
        assert finding.severity is RegressionSeverity.SEVERE
        assert finding.is_significant

    def test_a_vanishing_baseline_gives_the_same_verdict(self) -> None:
        tiny = TaskBaseline(task_id="t")
        tiny.add_metric("pnl", 1e-300, std=1.0, sample_size=20)
        finding = RegressionDetector().compare(tiny, [{"pnl": v} for v in DECLINE]).regressions[0]
        zero = _decline_report().regressions[0]
        assert (finding.severity, finding.is_significant) == (zero.severity, zero.is_significant)

    def test_an_improvement_from_zero_is_an_improvement_without_a_percentage(self) -> None:
        report = RegressionDetector().compare(
            _pnl_baseline(), [{"pnl": v} for v in [0.05, 0.04, 0.06, 0.05] * 5]
        )
        assert report.regressions == []
        finding = report.improvements[0]
        assert finding.delta == pytest.approx(0.05)
        assert finding.delta_percent is None

    def test_every_move_from_zero_is_reported_because_no_floor_can_apply(self) -> None:
        detector = RegressionDetector(min_delta_percent=50.0)
        report = detector.compare(_pnl_baseline(), [{"pnl": -0.01}] * 5)
        assert [r.metric_name for r in report.regressions] == ["pnl"]

    def test_no_move_from_zero_is_not_a_finding(self) -> None:
        report = RegressionDetector().compare(_pnl_baseline(), [{"pnl": 0.0}] * 5)
        assert report.regressions == [] and report.improvements == []

    def test_a_finding_without_a_percentage_round_trips_through_json(self) -> None:
        finding = _decline_report().regressions[0]
        data = json.loads(json.dumps(finding.model_dump(mode="json"), allow_nan=False))
        assert data["delta_percent"] is None and data["severity"] == "severe"
        restored = MetricRegression.model_validate(data)
        assert restored.delta_percent is None
        assert restored.severity is RegressionSeverity.SEVERE


class TestReportDecision:
    @pytest.mark.parametrize("threshold", list(RegressionSeverity)[1:])
    def test_a_significant_decline_from_zero_blocks_at_every_threshold(
        self, threshold: RegressionSeverity
    ) -> None:
        report = _decline_report()
        assert report.has_regression
        assert report.overall_severity is RegressionSeverity.SEVERE
        assert report.should_block_ci(threshold) is True

    def test_a_decline_from_zero_that_is_not_significant_does_not_block(self) -> None:
        report = RegressionDetector().compare(
            _pnl_baseline(), [{"pnl": -0.1}, {"pnl": 0.1}, {"pnl": -0.2}]
        )
        assert report.regressions and not report.regressions[0].is_significant
        assert report.should_block_ci() is False

    def test_text_says_unavailable_instead_of_inventing_a_percentage(self) -> None:
        report = _decline_report()
        for text in (report.to_ci_output(), report.summary):
            assert "+100.0%" not in text
            assert "N/A relative to 0" in text
        assert "REGRESSION DETECTED [SEVERE]" in report.to_ci_output()


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
    @pytest.mark.parametrize("threshold", list(RegressionSeverity)[1:])
    def test_a_significant_rise_from_zero_blocks_at_every_threshold(
        self, tmp_path: Path, threshold: RegressionSeverity
    ) -> None:
        gate = evaluate_gate(
            _batch({"cost": COST_RISE}), _cost_manager(tmp_path), threshold=threshold
        )
        task = gate.tasks[0]
        finding = next(r for r in task.regressions if r.metric_name == "mean_score")
        assert finding.is_significant and finding.delta_percent is None
        assert finding.severity is RegressionSeverity.SEVERE
        assert task.blocking and task.overall_severity is RegressionSeverity.SEVERE
        assert gate.status is GateStatus.BLOCKED and gate.exit_code == 1
        reason = next(r for r in gate.reasons if "blocking regression(s)" in r)
        assert "cost (severe)" in reason

    def test_the_decision_survives_a_round_trip(self, tmp_path: Path) -> None:
        gate = evaluate_gate(_batch({"cost": COST_RISE}), _cost_manager(tmp_path))
        data = json.loads(json.dumps(gate.to_dict(), allow_nan=False))
        assert data["tasks"][0]["overall_severity"] == "severe"
        restored = type(gate).from_dict(data)
        assert restored.status is GateStatus.BLOCKED
        finding = next(r for r in restored.tasks[0].regressions if r.metric_name == "mean_score")
        assert finding.delta_percent is None
        assert finding.severity is RegressionSeverity.SEVERE

    def test_reports_render_unavailable_not_plus_100(self, tmp_path: Path) -> None:
        batch = _batch({"cost": COST_RISE})
        gen = ReportGenerator(k_values=[1], consistency_k_values=[2])
        report = gen.build_report(batch)
        report.gate = evaluate_gate(batch, _cost_manager(tmp_path))
        rows = {row[1]: row for row in _gate_rows(report.gate)}
        assert rows["mean_score"][4] == "N/A relative to 0"
        assert rows["mean_score"][5] == "severe"
        assert "blocking" in rows["mean_score"][7]
        for text in (
            gen.render_markdown(report), gen.render_html(report), gen.render_ci_summary(report)
        ):
            assert "+100.0%" not in text
            assert "N/A relative to 0" in text

    def test_legacy_reports_render_a_finding_without_a_percentage(self) -> None:
        data = ReportData(regression_report=_decline_report())
        html = ReportGenerator().render_html(data)
        assert "+100.0%" not in html
        assert "<td>N/A relative to 0</td>" in html and "<td>severe</td>" in html
        assert "+100.0%" not in ReportGenerator().render_ci_summary(data)


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
    def test_a_suite_decline_from_a_zero_mean_has_no_percentage_and_blocks(self) -> None:
        result = _suite(SUITE_DECLINE, RegressionSeverity.SEVERE)
        assert result.baseline_mean == 0.0 and result.delta < 0
        assert result.delta_percent is None
        assert result.severity is RegressionSeverity.SEVERE
        assert result.is_regression and result.is_significant and result.blocking
        text = result.describe()
        assert "N/A relative to 0" in text and "+100.0%" not in text

    def test_the_suite_result_round_trips(self) -> None:
        result = _suite(SUITE_DECLINE, RegressionSeverity.MODERATE)
        data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
        assert data["delta_percent"] is None and data["severity"] == "severe"
        restored = SuiteGateResult.from_dict(data)
        assert restored.delta_percent is None
        assert restored.severity is RegressionSeverity.SEVERE
