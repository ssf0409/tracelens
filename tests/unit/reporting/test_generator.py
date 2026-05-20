"""Tests for report generator module."""

import pytest

from tracelens.core.outcome import Outcome
from tracelens.core.trial import Trial, TrialBatch, TrialStatus
from tracelens.reporting.generator import (
    ReportData,
    ReportGenerator,
    TaskSummary,
    _html_card,
    _pass_rate_color,
    _svg_bar_chart,
    _svg_histogram,
)


def _make_batch(task_configs: dict[str, list[tuple[bool, float]]]) -> TrialBatch:
    """Build a TrialBatch from {task_id: [(passed, score), ...]}."""
    batch = TrialBatch()
    for task_id, runs in task_configs.items():
        for i, (passed, score) in enumerate(runs):
            trial = Trial(
                task_id=task_id,
                run_index=i,
                total_runs=len(runs),
                status=TrialStatus.COMPLETED,
            )
            trial.add_outcome(Outcome(
                trial_id=trial.trial_id,
                grader_id="test_grader",
                passed=passed,
                score=score,
            ))
            batch.add_trial(trial)
    return batch


class TestTaskSummary:
    def test_to_dict(self):
        s = TaskSummary(
            task_id="t1", num_trials=5, pass_rate=0.8,
            mean_score=0.75, std_score=0.1,
        )
        d = s.to_dict()
        assert d["task_id"] == "t1"
        assert d["pass_rate"] == 0.8


class TestReportData:
    def test_to_dict_and_from_dict(self):
        report = ReportData(
            total_trials=10,
            total_tasks=2,
            overall_pass_rate=0.7,
            overall_mean_score=0.65,
            task_summaries=[
                TaskSummary(
                    task_id="t1", num_trials=5, pass_rate=0.8,
                    mean_score=0.75, std_score=0.1,
                ),
            ],
            pass_at_k={"pass@1": 0.7, "pass@3": 0.9},
            reliability={"pass^2": 0.6},
        )
        d = report.to_dict()
        restored = ReportData.from_dict(d)

        assert restored.total_trials == 10
        assert restored.total_tasks == 2
        assert restored.overall_pass_rate == 0.7
        assert len(restored.task_summaries) == 1
        assert restored.task_summaries[0].task_id == "t1"
        assert restored.pass_at_k["pass@1"] == 0.7

    def test_to_dict_without_regression(self):
        report = ReportData()
        d = report.to_dict()
        assert "regression" not in d


class TestReportGenerator:
    def test_build_report_basic(self):
        """Report contains correct suite-level stats."""
        batch = _make_batch({
            "t1": [(True, 0.9), (True, 0.8), (False, 0.3)],
            "t2": [(True, 0.7), (True, 0.6)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)

        assert report.total_trials == 5
        assert report.total_tasks == 2
        assert report.overall_pass_rate == pytest.approx(4 / 5)
        assert len(report.task_summaries) == 2

    def test_build_report_per_task(self):
        """Per-task summaries are computed correctly."""
        batch = _make_batch({
            "t1": [(True, 0.9), (False, 0.4)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)

        summary = report.task_summaries[0]
        assert summary.task_id == "t1"
        assert summary.num_trials == 2
        assert summary.pass_rate == pytest.approx(0.5)
        assert summary.mean_score == pytest.approx(0.65)

    def test_build_report_pass_at_k(self):
        """Suite-level pass@k is computed."""
        batch = _make_batch({
            "t1": [(True, 0.9), (True, 0.8), (True, 0.7)],
        })

        gen = ReportGenerator(k_values=[1, 3])
        report = gen.build_report(batch)

        assert "pass@1" in report.pass_at_k
        assert "pass@3" in report.pass_at_k
        assert report.pass_at_k["pass@1"] == pytest.approx(1.0)

    def test_render_markdown(self):
        """Markdown report contains expected sections."""
        batch = _make_batch({
            "t1": [(True, 0.9), (False, 0.4)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)
        md = gen.render_markdown(report)

        assert "# Evaluation Report" in md
        assert "## Summary" in md
        assert "Pass Rate" in md
        assert "t1" in md

    def test_render_ci_summary(self):
        """CI summary is a compact single line."""
        batch = _make_batch({
            "t1": [(True, 0.9)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)
        ci = gen.render_ci_summary(report)

        assert "TraceLens" in ci
        assert "pass_rate=" in ci

    def test_empty_batch(self):
        """Report handles empty batch gracefully."""
        batch = TrialBatch()
        gen = ReportGenerator()
        report = gen.build_report(batch)

        assert report.total_trials == 0
        assert report.total_tasks == 0
        assert report.overall_pass_rate == 0.0

    def test_render_html_basic(self):
        """HTML report is a valid self-contained document."""
        batch = _make_batch({
            "t1": [(True, 0.9), (True, 0.8), (False, 0.3)],
            "t2": [(True, 0.7), (True, 0.6)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)
        html = gen.render_html(report)

        assert "<!DOCTYPE html>" in html
        assert "<title>TraceLens Report</title>" in html
        assert "TraceLens v" in html

    def test_render_html_contains_summary_cards(self):
        """HTML report has summary cards with correct values."""
        batch = _make_batch({
            "t1": [(True, 0.9), (False, 0.4)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)
        html = gen.render_html(report)

        assert "Tasks" in html
        assert "Trials" in html
        assert "Pass Rate" in html
        assert "Mean Score" in html

    def test_render_html_contains_task_table(self):
        """HTML report has per-task results table."""
        batch = _make_batch({
            "task-alpha": [(True, 0.9)],
            "task-beta": [(False, 0.3)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)
        html = gen.render_html(report)

        assert "task-alpha" in html
        assert "task-beta" in html
        assert "Per-Task Results" in html

    def test_render_html_contains_svg_charts(self):
        """HTML report contains SVG chart elements."""
        batch = _make_batch({
            "t1": [(True, 0.9), (True, 0.8), (False, 0.3)],
        })

        gen = ReportGenerator(k_values=[1, 3])
        report = gen.build_report(batch)
        html = gen.render_html(report)

        assert "<svg" in html
        assert "pass@1" in html

    def test_render_html_empty_report(self):
        """HTML report handles empty data gracefully."""
        gen = ReportGenerator()
        report = ReportData()
        html = gen.render_html(report)

        assert "<!DOCTYPE html>" in html
        assert "0" in html  # zero tasks/trials

    def test_render_html_escapes_task_ids(self):
        """HTML report escapes special characters in task IDs."""
        batch = _make_batch({
            "task<script>": [(True, 0.9)],
        })

        gen = ReportGenerator()
        report = gen.build_report(batch)
        html = gen.render_html(report)

        assert "task<script>" not in html
        assert "task&lt;script&gt;" in html


class TestSvgHelpers:
    def test_pass_rate_color_green(self):
        assert _pass_rate_color(0.9) == "#22c55e"
        assert _pass_rate_color(0.8) == "#22c55e"

    def test_pass_rate_color_yellow(self):
        assert _pass_rate_color(0.6) == "#eab308"
        assert _pass_rate_color(0.5) == "#eab308"

    def test_pass_rate_color_red(self):
        assert _pass_rate_color(0.3) == "#ef4444"
        assert _pass_rate_color(0.0) == "#ef4444"

    def test_html_card(self):
        card = _html_card("Tasks", "5", "#3b82f6")
        assert "Tasks" in card
        assert "5" in card
        assert "#3b82f6" in card

    def test_svg_bar_chart_basic(self):
        svg = _svg_bar_chart(["a", "b"], [0.5, 0.8], 1.0)
        assert "<svg" in svg
        assert "a" in svg
        assert "b" in svg

    def test_svg_bar_chart_empty(self):
        assert _svg_bar_chart([], [], 1.0) == ""

    def test_svg_bar_chart_custom_colors(self):
        svg = _svg_bar_chart(["x"], [0.5], 1.0, ["#ff0000"])
        assert "#ff0000" in svg

    def test_svg_histogram_basic(self):
        svg = _svg_histogram([0.1, 0.3, 0.5, 0.7, 0.9], bins=5)
        assert "<svg" in svg
        assert "<rect" in svg

    def test_svg_histogram_empty(self):
        assert _svg_histogram([]) == ""

    def test_svg_histogram_single_value(self):
        svg = _svg_histogram([0.5, 0.5, 0.5])
        assert "<svg" in svg


class TestInfraErrorReporting:
    """Reports should surface infra-error rate alongside pass rate so
    infrastructure-driven failures aren't conflated with capability
    regressions (Anthropic, Feb 2026)."""

    def _batch_with_infra_errors(self):
        from tracelens.core.outcome import Outcome
        from tracelens.core.trial import Trial, TrialBatch, TrialStatus
        trials = [
            Trial(task_id="t1", status=TrialStatus.COMPLETED, outcomes=[
                Outcome(trial_id="x", grader_id="g", passed=True, score=1.0),
            ]),
            Trial(task_id="t1", status=TrialStatus.INFRA_ERROR),
            Trial(task_id="t2", status=TrialStatus.COMPLETED, outcomes=[
                Outcome(trial_id="y", grader_id="g", passed=False, score=0.1),
            ]),
            Trial(task_id="t2", status=TrialStatus.INFRA_ERROR),
        ]
        return TrialBatch(trials=trials)

    def test_build_report_populates_infra_metrics(self):
        from tracelens.reporting.generator import ReportGenerator

        report = ReportGenerator().build_report(self._batch_with_infra_errors())

        assert report.infra_error_count == 2
        assert report.infra_error_rate == 0.5

    def test_markdown_surfaces_infra_warning_when_rate_positive(self):
        from tracelens.reporting.generator import ReportGenerator

        gen = ReportGenerator()
        report = gen.build_report(self._batch_with_infra_errors())
        md = gen.render_markdown(report)

        assert "Infra-Error Rate" in md
        assert "50.0%" in md
        # The explanation is what gives readers context; ensure it appears.
        assert "infrastructure" in md.lower()

    def test_markdown_omits_infra_section_when_zero(self):
        """No noise = no section. Don't clutter the default report."""
        from tracelens.core.outcome import Outcome
        from tracelens.core.trial import Trial, TrialBatch, TrialStatus
        from tracelens.reporting.generator import ReportGenerator

        clean_batch = TrialBatch(trials=[
            Trial(task_id="t1", status=TrialStatus.COMPLETED, outcomes=[
                Outcome(trial_id="x", grader_id="g", passed=True, score=1.0),
            ]),
        ])
        gen = ReportGenerator()
        md = gen.render_markdown(gen.build_report(clean_batch))
        assert "Infra-Error Rate" not in md

    def test_ci_summary_includes_infra_errors_when_nonzero(self):
        from tracelens.reporting.generator import ReportGenerator

        gen = ReportGenerator()
        report = gen.build_report(self._batch_with_infra_errors())
        ci = gen.render_ci_summary(report)

        assert "infra_errors=50.0%" in ci

    def test_to_dict_roundtrip_preserves_infra_fields(self):
        from tracelens.reporting.generator import ReportData, ReportGenerator

        gen = ReportGenerator()
        original = gen.build_report(self._batch_with_infra_errors())
        restored = ReportData.from_dict(original.to_dict())

        assert restored.infra_error_count == 2
        assert restored.infra_error_rate == 0.5
