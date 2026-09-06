"""Tests for report generator module."""

import json

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
        # Distinct run indices per task: duplicates are rejected (#45).
        trials = [
            Trial(task_id="t1", run_index=0, status=TrialStatus.COMPLETED, outcomes=[
                Outcome(trial_id="x", grader_id="g", passed=True, score=1.0),
            ]),
            Trial(task_id="t1", run_index=1, status=TrialStatus.INFRA_ERROR),
            Trial(task_id="t2", run_index=0, status=TrialStatus.COMPLETED, outcomes=[
                Outcome(trial_id="y", grader_id="g", passed=False, score=0.1),
            ]),
            Trial(task_id="t2", run_index=1, status=TrialStatus.INFRA_ERROR),
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


class TestReportRunOrderIndependence:
    """Issue #45: identical (task, run_index, outcome) data give identical reports."""

    @staticmethod
    def _batch_in_order(order: list[int]) -> TrialBatch:
        outcomes = {0: True, 1: True, 2: False, 3: False}
        batch = TrialBatch()
        for run_index in order:
            trial = Trial(
                task_id="t", run_index=run_index, total_runs=4, status=TrialStatus.COMPLETED
            )
            trial.add_outcome(Outcome(
                trial_id=trial.trial_id,
                grader_id="g",
                passed=outcomes[run_index],
                score=1.0 if outcomes[run_index] else 0.0,
            ))
            batch.add_trial(trial)
        return batch

    def test_pass_hat_k_ignores_completion_order(self):
        gen = ReportGenerator(k_values=[1], consistency_k_values=[2])
        in_order = gen.build_report(self._batch_in_order([0, 1, 2, 3]))
        reordered = gen.build_report(self._batch_in_order([0, 2, 3, 1]))

        # run_index order T T F F -> windows TT, TF, FF -> 1/3.
        # Completion order T F F T would have given 0.
        assert in_order.reliability["pass^2"] == pytest.approx(1 / 3)
        assert reordered.reliability["pass^2"] == pytest.approx(1 / 3)
        assert reordered.pass_at_k == in_order.pass_at_k
        assert reordered.task_summaries[0].to_dict() == in_order.task_summaries[0].to_dict()

    def test_missing_run_does_not_bridge_windows(self):
        batch = TrialBatch()
        for run_index, passed in [(0, True), (2, True)]:  # run 1 missing
            trial = Trial(task_id="t", run_index=run_index, status=TrialStatus.COMPLETED)
            trial.add_outcome(Outcome(
                trial_id=trial.trial_id, grader_id="g", passed=passed, score=1.0
            ))
            batch.add_trial(trial)
        report = ReportGenerator(k_values=[1], consistency_k_values=[2]).build_report(batch)
        # No complete window of 2: the task is ineligible, so suite pass^2 is
        # unavailable rather than the 1.0 that treating runs 0 and 2 as
        # consecutive would give.
        assert report.reliability["pass^2"] is None
        assert not report.metric_availability["pass^2"].available
        assert report.task_summaries[0].pass_rate == 1.0


class TestMetricAvailabilityReporting:
    """Issue #46: unavailable metrics render as N/A with a reason, never as zeros."""

    @staticmethod
    def _one_run_suite() -> TrialBatch:
        return _make_batch({"t1": [(True, 1.0)], "t2": [(True, 1.0)]})

    def test_one_run_suite_marks_higher_k_unavailable(self):
        report = ReportGenerator().build_report(self._one_run_suite())
        assert report.pass_at_k["pass@1"] == 1.0
        assert report.pass_at_k["pass@3"] is None
        assert report.pass_at_k["pass@5"] is None
        assert all(v is None for v in report.reliability.values())
        mv = report.metric_availability["pass@5"]
        assert not mv.available
        assert (mv.eligible_tasks, mv.total_tasks, mv.max_runs, mv.required_runs) == (0, 2, 1, 5)
        assert report.metric_availability["pass@1"].available
        for summary in report.task_summaries:
            assert summary.pass_at_k["pass@1"] == 1.0
            assert summary.pass_at_k["pass@5"] is None
            assert summary.gradable_trials == 1

    def test_markdown_renders_na_with_reason_and_runs_hint(self):
        gen = ReportGenerator()
        md = gen.render_markdown(gen.build_report(self._one_run_suite()))
        assert "**pass@1**: 1.0000 (2/2 tasks)" in md
        assert (
            "**pass@5**: N/A: needs at least 5 gradable runs per task; "
            "0/2 tasks eligible; max 1 gradable run(s) recorded"
        ) in md
        assert "rerun with `--num-runs 5`" in md
        reliability_section = md.split("## Reliability (pass^k)")[1].split("## Per-Task")[0]
        assert "0.0000" not in reliability_section
        assert "N/A" in reliability_section

    def test_ci_summary_renders_na(self):
        gen = ReportGenerator()
        ci = gen.render_ci_summary(gen.build_report(self._one_run_suite()))
        assert "pass@1=1.0000" in ci
        assert "pass@3=n/a" in ci
        assert "pass@5=n/a" in ci

    def test_html_lists_unavailable_metrics_without_zero_bars(self):
        gen = ReportGenerator()
        html = gen.render_html(gen.build_report(self._one_run_suite()))
        assert "<strong>pass@5</strong>: N/A: needs at least 5 gradable runs per task" in html
        reliability_section = html.split("Reliability (pass^k)")[1].split("</section>")[0]
        assert "<rect" not in reliability_section  # nothing drawn as a zero-height bar
        assert "N/A" in reliability_section
        assert "--num-runs 5" in html

    def test_json_roundtrip_preserves_availability(self):
        gen = ReportGenerator()
        original = gen.build_report(self._one_run_suite())
        restored = ReportData.from_dict(json.loads(json.dumps(original.to_dict())))
        assert restored.availability_recorded is True
        assert restored.pass_at_k == original.pass_at_k
        assert restored.reliability == original.reliability
        assert restored.metric_availability == original.metric_availability
        assert restored.gradable_trials == original.gradable_trials == 2
        assert restored.task_summaries[0].gradable_trials == 1
        assert original.to_dict()["metric_availability"]["pass@5"]["available"] is False

    def test_legacy_report_loads_with_explicit_assumption(self):
        legacy = {
            "total_trials": 2, "total_tasks": 2,
            "overall_pass_rate": 1.0, "overall_mean_score": 1.0,
            "pass_at_k": {"pass@1": 1.0, "pass@5": 1.0},
            "reliability": {"pass^2": 0.0},
            "task_summaries": [{
                "task_id": "t1", "num_trials": 1, "pass_rate": 1.0,
                "mean_score": 1.0, "std_score": 0.0,
                "pass_at_k": {"pass@1": 1.0}, "reliability": {"pass^2": 0.0},
            }],
        }
        report = ReportData.from_dict(legacy)
        assert report.availability_recorded is False
        assert report.gradable_trials == 2  # legacy denominator: all trials
        assert report.pass_at_k["pass@5"] == 1.0  # shown as recorded
        assert report.metric_availability["pass^2"].total_tasks is None
        assert report.task_summaries[0].gradable_trials is None

        gen = ReportGenerator()
        md = gen.render_markdown(report)
        assert "availability was not recorded" in md
        assert "**pass^2**: 0.0000" in md
        assert "| t1 | 1 | 100.0% |" in md
        assert "pass_rate=100.0%" in gen.render_ci_summary(report)
        assert "availability was not recorded" in gen.render_html(report)

    @staticmethod
    def _batch_with_harness_failures() -> TrialBatch:
        def trial(task_id, run_index, status, passed=None, grader_error=False):
            t = Trial(task_id=task_id, run_index=run_index, status=status)
            if passed is not None:
                t.add_outcome(Outcome(
                    trial_id=t.trial_id, grader_id="g", passed=passed,
                    score=1.0 if passed else 0.0, grader_error=grader_error,
                ))
            return t

        return TrialBatch(trials=[
            trial("t1", 0, TrialStatus.COMPLETED, passed=True),
            trial("t1", 1, TrialStatus.INFRA_ERROR),
            trial("t2", 0, TrialStatus.COMPLETED, passed=False),
            trial("t2", 1, TrialStatus.COMPLETED, passed=False, grader_error=True),
            trial("t3", 0, TrialStatus.SKIPPED),
        ])

    def test_harness_failures_are_excluded_and_counted(self):
        gen = ReportGenerator(k_values=[1], consistency_k_values=[2])
        report = gen.build_report(self._batch_with_harness_failures())
        assert report.total_trials == 5
        assert report.gradable_trials == 2
        assert report.excluded_trials == 3
        assert report.overall_pass_rate == 0.5
        assert report.infra_error_count == 1
        assert report.grader_error_count == 1
        mv = report.metric_availability["pass@1"]
        assert (mv.eligible_tasks, mv.total_tasks) == (2, 3)
        by_task = {s.task_id: s for s in report.task_summaries}
        assert (by_task["t1"].num_trials, by_task["t1"].gradable_trials) == (2, 1)
        assert by_task["t3"].gradable_trials == 0

        md = gen.render_markdown(report)
        assert "50.0% (over 2 gradable trials; 3 excluded as harness failures or never run)" in md
        assert "| t1 | 2 (1 gradable) | 100.0% |" in md
        assert "| t3 | 1 (0 gradable) | N/A |" in md

    def test_no_gradable_trials_renders_na_pass_rate(self):
        batch = TrialBatch(trials=[
            Trial(task_id="t1", run_index=0, status=TrialStatus.INFRA_ERROR),
            Trial(task_id="t1", run_index=1, status=TrialStatus.INFRA_ERROR),
        ])
        gen = ReportGenerator(k_values=[1], consistency_k_values=[2])
        report = gen.build_report(batch)
        assert report.gradable_trials == 0
        assert report.pass_at_k["pass@1"] is None
        assert report.metric_availability["pass@1"].reason == "needs at least 1 gradable runs per task"
        assert "**Pass Rate**: N/A (no gradable trials)" in gen.render_markdown(report)
        assert "pass_rate=n/a" in gen.render_ci_summary(report)
        assert ">N/A<" in gen.render_html(report)
