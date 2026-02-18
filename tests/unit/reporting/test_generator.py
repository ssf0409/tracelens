"""Tests for report generator module."""

import pytest

from eval_kit.core.outcome import Outcome
from eval_kit.core.trial import Trial, TrialBatch, TrialStatus
from eval_kit.reporting.generator import ReportData, ReportGenerator, TaskSummary


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

        assert "eval-kit" in ci
        assert "pass_rate=" in ci

    def test_empty_batch(self):
        """Report handles empty batch gracefully."""
        batch = TrialBatch()
        gen = ReportGenerator()
        report = gen.build_report(batch)

        assert report.total_trials == 0
        assert report.total_tasks == 0
        assert report.overall_pass_rate == 0.0
