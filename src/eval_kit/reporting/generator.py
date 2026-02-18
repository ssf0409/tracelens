"""Report generation for evaluation results.

Transforms TrialBatch data into structured reports with per-task
summaries, statistical analysis, and optional regression detection.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from eval_kit.baselines.comparison import RegressionReport
from eval_kit.baselines.manager import BaselineManager
from eval_kit.core.trial import TrialBatch
from eval_kit.statistics.pass_at_k import pass_at_k, PassAtKAnalyzer
from eval_kit.statistics.consistency import ConsistencyAnalyzer


@dataclass
class TaskSummary:
    """Per-task summary statistics."""

    task_id: str
    num_trials: int
    pass_rate: float
    mean_score: float
    std_score: float
    pass_at_k: dict[str, float] = field(default_factory=dict)
    reliability: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "num_trials": self.num_trials,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "pass_at_k": self.pass_at_k,
            "reliability": self.reliability,
        }


@dataclass
class ReportData:
    """Complete evaluation report."""

    # Suite-level stats
    total_trials: int = 0
    total_tasks: int = 0
    overall_pass_rate: float = 0.0
    overall_mean_score: float = 0.0

    # Per-task summaries
    task_summaries: list[TaskSummary] = field(default_factory=list)

    # Statistical analysis
    pass_at_k: dict[str, float] = field(default_factory=dict)
    reliability: dict[str, float] = field(default_factory=dict)

    # Optional regression report
    regression_report: RegressionReport | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_trials": self.total_trials,
            "total_tasks": self.total_tasks,
            "overall_pass_rate": self.overall_pass_rate,
            "overall_mean_score": self.overall_mean_score,
            "pass_at_k": self.pass_at_k,
            "reliability": self.reliability,
            "task_summaries": [s.to_dict() for s in self.task_summaries],
        }
        if self.regression_report:
            result["regression"] = {
                "has_regression": self.regression_report.has_regression,
                "severity": self.regression_report.overall_severity.value,
                "summary": self.regression_report.summary,
            }
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReportData":
        summaries = [
            TaskSummary(**s) for s in data.get("task_summaries", [])
        ]
        return cls(
            total_trials=data.get("total_trials", 0),
            total_tasks=data.get("total_tasks", 0),
            overall_pass_rate=data.get("overall_pass_rate", 0.0),
            overall_mean_score=data.get("overall_mean_score", 0.0),
            task_summaries=summaries,
            pass_at_k=data.get("pass_at_k", {}),
            reliability=data.get("reliability", {}),
        )


class ReportGenerator:
    """Generates evaluation reports from TrialBatch results.

    Example:
        gen = ReportGenerator()
        report = gen.build_report(batch)
        print(gen.render_markdown(report))
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        consistency_k_values: list[int] | None = None,
    ) -> None:
        self._pass_at_k = PassAtKAnalyzer(k_values=k_values or [1, 3, 5])
        self._consistency = ConsistencyAnalyzer(
            k_values=consistency_k_values or [2, 3, 5]
        )

    def build_report(
        self,
        batch: TrialBatch,
        baseline_manager: BaselineManager | None = None,
    ) -> ReportData:
        """Build a ReportData from a TrialBatch."""
        pass_results = batch.get_pass_results_by_task()
        task_ids = sorted(pass_results.keys())

        # Suite-level stats
        all_scores = []
        task_summaries = []

        for task_id in task_ids:
            trials = batch.get_trials_for_task(task_id)
            passes = pass_results[task_id]

            scores = []
            for trial in trials:
                if trial.aggregate_score is not None:
                    scores.append(trial.aggregate_score)

            pass_rate = sum(passes) / len(passes) if passes else 0.0
            mean_score = float(np.mean(scores)) if scores else 0.0
            std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

            # Per-task pass@k
            task_pass_at_k = self._pass_at_k.analyze({task_id: passes})
            task_reliability = self._consistency.analyze({task_id: passes})

            task_summaries.append(TaskSummary(
                task_id=task_id,
                num_trials=len(trials),
                pass_rate=pass_rate,
                mean_score=mean_score,
                std_score=std_score,
                pass_at_k=task_pass_at_k,
                reliability=task_reliability,
            ))

            all_scores.extend(scores)

        # Suite-level pass@k and reliability
        suite_pass_at_k = self._pass_at_k.analyze(pass_results)
        suite_reliability = self._consistency.analyze(pass_results)

        report = ReportData(
            total_trials=batch.total_count,
            total_tasks=len(task_ids),
            overall_pass_rate=batch.pass_rate,
            overall_mean_score=float(np.mean(all_scores)) if all_scores else 0.0,
            task_summaries=task_summaries,
            pass_at_k=suite_pass_at_k,
            reliability=suite_reliability,
        )

        return report

    def render_markdown(self, report: ReportData) -> str:
        """Render a human-readable markdown report."""
        lines = [
            "# Evaluation Report",
            "",
            "## Summary",
            "",
            f"- **Tasks**: {report.total_tasks}",
            f"- **Trials**: {report.total_trials}",
            f"- **Pass Rate**: {report.overall_pass_rate:.1%}",
            f"- **Mean Score**: {report.overall_mean_score:.4f}",
            "",
        ]

        # pass@k summary
        if report.pass_at_k:
            lines.append("## Capability (pass@k)")
            lines.append("")
            for key, val in sorted(report.pass_at_k.items()):
                lines.append(f"- **{key}**: {val:.4f}")
            lines.append("")

        # Reliability summary
        if report.reliability:
            lines.append("## Reliability (pass^k)")
            lines.append("")
            for key, val in sorted(report.reliability.items()):
                lines.append(f"- **{key}**: {val:.4f}")
            lines.append("")

        # Per-task table
        if report.task_summaries:
            lines.append("## Per-Task Results")
            lines.append("")
            lines.append("| Task | Trials | Pass Rate | Mean Score |")
            lines.append("|------|--------|-----------|------------|")
            for s in report.task_summaries:
                lines.append(
                    f"| {s.task_id} | {s.num_trials} | "
                    f"{s.pass_rate:.1%} | {s.mean_score:.4f} |"
                )
            lines.append("")

        # Regression
        if report.regression_report and report.regression_report.has_regression:
            lines.append("## Regression Alert")
            lines.append("")
            lines.append(report.regression_report.to_ci_output())
            lines.append("")

        return "\n".join(lines)

    def render_ci_summary(self, report: ReportData) -> str:
        """Render a compact CI-friendly summary."""
        lines = [
            f"eval-kit: {report.total_tasks} tasks, "
            f"{report.total_trials} trials, "
            f"pass_rate={report.overall_pass_rate:.1%}, "
            f"mean_score={report.overall_mean_score:.4f}",
        ]

        for key, val in sorted(report.pass_at_k.items()):
            lines[0] += f", {key}={val:.4f}"

        if report.regression_report and report.regression_report.has_regression:
            lines.append(
                f"REGRESSION [{report.regression_report.overall_severity.value.upper()}]"
            )

        return "\n".join(lines)
