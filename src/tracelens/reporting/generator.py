"""Report generation for evaluation results.

Transforms TrialBatch data into structured reports with per-task
summaries, statistical analysis, and optional regression detection.
Supports markdown, CI summary, and self-contained HTML dashboard output.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import escape
from typing import Any

import numpy as np

from tracelens._version import __version__
from tracelens.baselines.comparison import RegressionReport
from tracelens.baselines.manager import BaselineManager
from tracelens.core.trial import TrialBatch
from tracelens.statistics.consistency import ConsistencyAnalyzer
from tracelens.statistics.pass_at_k import PassAtKAnalyzer


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

    # Infrastructure-noise awareness (see Anthropic, Feb 2026).
    # ``infra_error_rate`` is reported alongside pass_rate so a spike
    # signals "re-check the eval config" rather than "model got worse."
    infra_error_count: int = 0
    infra_error_rate: float = 0.0

    # Grader-harness health: trials where a grader crashed and a failed
    # outcome was synthesized. A spike here means the grading harness is
    # broken, not that the agent regressed.
    grader_error_count: int = 0
    grader_error_rate: float = 0.0

    # Token usage across all trial transcripts — cost visibility for
    # LLM-heavy evals without walking every transcript by hand.
    total_input_tokens: int = 0
    total_output_tokens: int = 0

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
            "infra_error_count": self.infra_error_count,
            "infra_error_rate": self.infra_error_rate,
            "grader_error_count": self.grader_error_count,
            "grader_error_rate": self.grader_error_rate,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "pass_at_k": self.pass_at_k,
            "reliability": self.reliability,
            "task_summaries": [s.to_dict() for s in self.task_summaries],
        }
        if self.regression_report:
            result["regression"] = {
                "has_regression": self.regression_report.has_regression,
                "severity": self.regression_report.overall_severity.value,
                "summary": self.regression_report.summary,
                "infra_config_mismatch": self.regression_report.infra_config_mismatch,
                "blocking_regressions": len(self.regression_report.blocking_regressions),
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
            infra_error_count=data.get("infra_error_count", 0),
            infra_error_rate=data.get("infra_error_rate", 0.0),
            grader_error_count=data.get("grader_error_count", 0),
            grader_error_rate=data.get("grader_error_rate", 0.0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
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
            infra_error_count=batch.infra_error_count,
            infra_error_rate=batch.infra_error_rate,
            grader_error_count=batch.grader_error_count,
            grader_error_rate=batch.grader_error_rate,
            total_input_tokens=batch.total_input_tokens,
            total_output_tokens=batch.total_output_tokens,
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
        ]
        if report.infra_error_count > 0:
            lines.append(
                f"- **Infra-Error Rate**: {report.infra_error_rate:.1%} "
                f"({report.infra_error_count} of {report.total_trials} trials)"
            )
            lines.append(
                "  - ⚠ A non-zero infra-error rate means some trials failed "
                "due to infrastructure (OOM, network, sandbox) rather than "
                "the agent. Cross-check resource config before blaming the model."
            )
        lines.append("")

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
            f"TraceLens: {report.total_tasks} tasks, "
            f"{report.total_trials} trials, "
            f"pass_rate={report.overall_pass_rate:.1%}, "
            f"mean_score={report.overall_mean_score:.4f}",
        ]

        for key, val in sorted(report.pass_at_k.items()):
            lines[0] += f", {key}={val:.4f}"

        if report.infra_error_count > 0:
            lines[0] += f", infra_errors={report.infra_error_rate:.1%}"

        if report.regression_report and report.regression_report.has_regression:
            # Prefer the noise-aware "blocking" count if specs were provided.
            n_blocking = len(report.regression_report.blocking_regressions)
            n_total = len(report.regression_report.regressions)
            severity = report.regression_report.overall_severity.value.upper()
            if report.regression_report.infra_config_mismatch and n_blocking < n_total:
                lines.append(
                    f"REGRESSION [{severity}] — {n_blocking}/{n_total} blocking "
                    f"({n_total - n_blocking} within infra-noise band; configs differ)"
                )
            else:
                lines.append(f"REGRESSION [{severity}]")

        return "\n".join(lines)

    def render_html(self, report: ReportData) -> str:
        """Render a self-contained HTML dashboard with inline CSS and SVG charts."""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        pass_rate_pct = report.overall_pass_rate * 100
        pass_rate_color = _pass_rate_color(report.overall_pass_rate)

        # --- Summary cards ---
        cards_html = (
            _html_card("Tasks", str(report.total_tasks), "#3b82f6")
            + _html_card("Trials", str(report.total_trials), "#6366f1")
            + _html_card("Pass Rate", f"{pass_rate_pct:.1f}%", pass_rate_color)
            + _html_card("Mean Score", f"{report.overall_mean_score:.4f}", "#8b5cf6")
        )
        # Only surface the infra-error card when there's something to
        # see — zero infra errors shouldn't clutter the dashboard.
        if report.infra_error_count > 0:
            infra_color = "#ef4444" if report.infra_error_rate >= 0.05 else "#f97316"
            cards_html += _html_card(
                "Infra Errors",
                f"{report.infra_error_rate * 100:.1f}%",
                infra_color,
            )

        # --- Capability / Reliability bar charts ---
        capability_svg = ""
        if report.pass_at_k:
            labels = sorted(report.pass_at_k.keys())
            values = [report.pass_at_k[k] for k in labels]
            capability_svg = _svg_bar_chart(labels, values, 1.0)

        reliability_svg = ""
        if report.reliability:
            labels = sorted(report.reliability.keys())
            values = [report.reliability[k] for k in labels]
            reliability_svg = _svg_bar_chart(labels, values, 1.0)

        # --- Per-task table ---
        task_rows = ""
        for s in report.task_summaries:
            pr_color = _pass_rate_color(s.pass_rate)
            task_rows += (
                f'<tr><td>{escape(s.task_id)}</td>'
                f"<td>{s.num_trials}</td>"
                f'<td style="color:{pr_color};font-weight:600">'
                f"{s.pass_rate:.1%}</td>"
                f"<td>{s.mean_score:.4f}</td>"
                f"<td>{s.std_score:.4f}</td></tr>\n"
            )

        # --- Pass rate distribution chart ---
        pass_rate_chart = ""
        if report.task_summaries:
            pr_labels = [s.task_id for s in report.task_summaries]
            pr_values = [s.pass_rate for s in report.task_summaries]
            pr_colors = [_pass_rate_color(v) for v in pr_values]
            pass_rate_chart = _svg_bar_chart(pr_labels, pr_values, 1.0, pr_colors)

        # --- Score distribution histogram ---
        all_scores = [
            s.mean_score for s in report.task_summaries if s.mean_score > 0
        ]
        score_histogram = ""
        if all_scores:
            score_histogram = _svg_histogram(all_scores, bins=10)

        # --- Regression alert ---
        regression_html = ""
        if report.regression_report and report.regression_report.has_regression:
            severity = report.regression_report.overall_severity.value.upper()
            sev_color = {"MINOR": "#eab308", "MODERATE": "#f97316", "SEVERE": "#ef4444"}.get(
                severity, "#6b7280"
            )
            reg_rows = ""
            for r in report.regression_report.regressions:
                reg_rows += (
                    f"<tr><td>{escape(r.metric_name)}</td>"
                    f"<td>{r.baseline_mean:.4f}</td>"
                    f"<td>{r.current_mean:.4f}</td>"
                    f"<td>{r.delta_percent:+.1f}%</td>"
                    f"<td>{r.severity.value}</td></tr>\n"
                )
            imp_rows = ""
            for i in report.regression_report.improvements:
                imp_rows += (
                    f"<tr><td>{escape(i.metric_name)}</td>"
                    f"<td>{i.baseline_mean:.4f}</td>"
                    f"<td>{i.current_mean:.4f}</td>"
                    f"<td>{i.delta_percent:+.1f}%</td></tr>\n"
                )
            regression_html = f"""
    <section>
      <h2>Regression Alert
        <span style="background:{sev_color};color:#fff;padding:2px 10px;
          border-radius:12px;font-size:0.75em;margin-left:8px">{severity}</span>
      </h2>
      <table><thead><tr>
        <th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th><th>Severity</th>
      </tr></thead><tbody>{reg_rows}</tbody></table>
      {"<h3>Improvements</h3><table><thead><tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr></thead><tbody>" + imp_rows + "</tbody></table>" if imp_rows else ""}
    </section>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraceLens Report</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0;
         padding: 20px; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 1.5em; margin: 0 0 4px; }}
  h2 {{ font-size: 1.15em; margin: 24px 0 12px; border-bottom: 1px solid #e2e8f0;
        padding-bottom: 4px; }}
  .subtitle {{ color: #64748b; font-size: 0.85em; margin-bottom: 16px; }}
  .cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }}
  .card {{ background: #fff; border-radius: 8px; padding: 14px 20px;
           box-shadow: 0 1px 3px rgba(0,0,0,.08); min-width: 140px; flex: 1; }}
  .card .label {{ font-size: 0.75em; color: #64748b; text-transform: uppercase;
                  letter-spacing: 0.05em; }}
  .card .value {{ font-size: 1.6em; font-weight: 700; margin-top: 2px; }}
  section {{ background: #fff; border-radius: 8px; padding: 16px 20px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid #e2e8f0;
        color: #475569; font-weight: 600; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #f1f5f9; }}
  tr:hover {{ background: #f8fafc; }}
  .chart-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .chart-row > div {{ flex: 1; min-width: 280px; }}
  svg text {{ font-family: system-ui, sans-serif; }}
  @media print {{ body {{ background: #fff; }} section {{ box-shadow: none;
    border: 1px solid #e2e8f0; }} }}
</style>
</head>
<body>
<h1>TraceLens Evaluation Report</h1>
<div class="subtitle">Generated {escape(timestamp)} &middot; TraceLens v{__version__}</div>

<div class="cards">{cards_html}</div>

{"<section><h2>Capability (pass@k)</h2>" + capability_svg + "</section>" if capability_svg else ""}
{"<section><h2>Reliability (pass^k)</h2>" + reliability_svg + "</section>" if reliability_svg else ""}

{"<section><h2>Per-Task Results</h2><table><thead><tr><th>Task</th><th>Trials</th><th>Pass Rate</th><th>Mean Score</th><th>Std Score</th></tr></thead><tbody>" + task_rows + "</tbody></table></section>" if task_rows else ""}

{"<section><h2>Pass Rate Distribution</h2>" + pass_rate_chart + "</section>" if pass_rate_chart else ""}

{"<section><h2>Score Distribution</h2>" + score_histogram + "</section>" if score_histogram else ""}

{regression_html}

<div class="subtitle" style="margin-top:24px;text-align:center">
  TraceLens v{__version__} &middot; {escape(timestamp)}
</div>
</body>
</html>"""


def _pass_rate_color(rate: float) -> str:
    """Return a color hex based on pass rate."""
    if rate >= 0.8:
        return "#22c55e"
    if rate >= 0.5:
        return "#eab308"
    return "#ef4444"


def _html_card(label: str, value: str, color: str) -> str:
    """Render a summary card."""
    return (
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value" style="color:{color}">{escape(value)}</div></div>'
    )


def _svg_bar_chart(
    labels: list[str],
    values: list[float],
    max_value: float,
    colors: list[str] | None = None,
) -> str:
    """Generate an inline SVG horizontal bar chart."""
    if not labels:
        return ""

    bar_height = 28
    gap = 6
    label_width = 100
    chart_width = 500
    total_height = len(labels) * (bar_height + gap) + 10

    default_color = "#3b82f6"
    if colors is None:
        colors = [default_color] * len(labels)

    bars = ""
    for i, (label, val, color) in enumerate(zip(labels, values, colors)):
        y = i * (bar_height + gap) + 5
        bar_w = max(2, (val / max_value) * (chart_width - label_width - 60)) if max_value > 0 else 2
        text_val = f"{val:.2f}" if val < 10 else f"{val:.0f}"

        bars += (
            f'<text x="0" y="{y + bar_height * 0.7}" '
            f'font-size="12" fill="#475569">{escape(label)}</text>'
            f'<rect x="{label_width}" y="{y}" width="{bar_w:.1f}" '
            f'height="{bar_height}" rx="4" fill="{color}" opacity="0.85"/>'
            f'<text x="{label_width + bar_w + 6:.1f}" y="{y + bar_height * 0.7}" '
            f'font-size="12" fill="#1e293b" font-weight="600">{text_val}</text>'
        )

    return (
        f'<svg width="100%" viewBox="0 0 {chart_width} {total_height}" '
        f'xmlns="http://www.w3.org/2000/svg">{bars}</svg>'
    )


def _svg_histogram(values: list[float], bins: int = 10) -> str:
    """Generate an inline SVG histogram."""
    if not values:
        return ""

    min_val = min(values)
    max_val = max(values)

    if min_val == max_val:
        counts = [len(values)]
        bin_edges = [min_val, max_val + 1]
    else:
        counts_arr, bin_edges_arr = np.histogram(values, bins=bins)
        counts = counts_arr.tolist()
        bin_edges = bin_edges_arr.tolist()

    max_count = max(counts) if counts else 1

    chart_width = 500
    chart_height = 200
    margin_bottom = 30
    margin_left = 40
    margin_top = 10
    bar_area_width = chart_width - margin_left - 10
    bar_area_height = chart_height - margin_bottom - margin_top

    bar_width = bar_area_width / len(counts)

    bars = ""
    for i, count in enumerate(counts):
        x = margin_left + i * bar_width
        h = (count / max_count) * bar_area_height if max_count > 0 else 0
        y = margin_top + bar_area_height - h
        color = _pass_rate_color(bin_edges[i] if i < len(bin_edges) else 0.5)

        bars += (
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{max(bar_width - 2, 1):.1f}" height="{h:.1f}" '
            f'rx="2" fill="{color}" opacity="0.85"/>'
        )

    # X-axis labels (first, middle, last)
    x_labels = ""
    for idx in [0, len(counts) // 2, len(counts)]:
        if idx < len(bin_edges):
            x_pos = margin_left + idx * bar_width
            x_labels += (
                f'<text x="{x_pos:.1f}" y="{chart_height - 5}" '
                f'font-size="11" fill="#64748b" text-anchor="middle">'
                f"{bin_edges[idx]:.2f}</text>"
            )

    # Y-axis labels
    y_labels = ""
    for frac in [0, 0.5, 1.0]:
        y_pos = margin_top + bar_area_height * (1 - frac)
        val = int(max_count * frac)
        y_labels += (
            f'<text x="{margin_left - 5}" y="{y_pos + 4:.1f}" '
            f'font-size="11" fill="#64748b" text-anchor="end">{val}</text>'
        )

    return (
        f'<svg width="100%" viewBox="0 0 {chart_width} {chart_height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f"{y_labels}{bars}{x_labels}</svg>"
    )
