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
from tracelens.core.provenance import RunProvenance
from tracelens.core.trial import TrialBatch
from tracelens.reporting.gate import GateResult, GateStatus, TaskGateOutcome
from tracelens.statistics.availability import MetricValue
from tracelens.statistics.consistency import ConsistencyAnalyzer
from tracelens.statistics.pass_at_k import PassAtKAnalyzer


@dataclass
class TaskSummary:
    """Per-task summary statistics.

    ``pass_at_k`` and ``reliability`` map metric names to values, with
    ``None`` where the metric is unavailable for this task (for example
    fewer gradable runs than ``k``). ``gradable_trials`` counts the trials
    that are agent evidence; ``None`` means a legacy report that did not
    record it.
    """

    task_id: str
    num_trials: int
    pass_rate: float
    mean_score: float
    std_score: float
    pass_at_k: dict[str, float | None] = field(default_factory=dict)
    reliability: dict[str, float | None] = field(default_factory=dict)
    gradable_trials: int | None = None
    # Content hash of the task from the run's provenance; ``None`` when the
    # batch carried none. Store it on a baseline so the gate can tell a
    # changed task from a regressed one.
    task_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "num_trials": self.num_trials,
            "gradable_trials": self.gradable_trials,
            "pass_rate": self.pass_rate,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "pass_at_k": self.pass_at_k,
            "reliability": self.reliability,
            "task_hash": self.task_hash,
        }


@dataclass
class ReportData:
    """Complete evaluation report.

    Suite metrics live in ``pass_at_k`` / ``reliability`` (``None`` where
    unavailable) with their evidence in ``metric_availability``. A report
    loaded from JSON written before availability was recorded has
    ``availability_recorded=False``: its values are shown as recorded, and
    a zero may be an unavailable metric rather than a measured zero.
    """

    # Suite-level stats
    total_trials: int = 0
    total_tasks: int = 0
    overall_pass_rate: float = 0.0
    overall_mean_score: float = 0.0

    # Trials that are agent evidence (``Trial.is_gradable``): the
    # denominator of ``overall_pass_rate``. Harness failures and trials
    # that never ran are ``total_trials - gradable_trials``.
    gradable_trials: int = 0

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

    # Statistical analysis (``None`` = unavailable; see metric_availability)
    pass_at_k: dict[str, float | None] = field(default_factory=dict)
    reliability: dict[str, float | None] = field(default_factory=dict)
    metric_availability: dict[str, MetricValue] = field(default_factory=dict)
    availability_recorded: bool = True

    # Optional regression report (library callers may attach one by hand;
    # CLI runs record the full decision in ``gate`` instead)
    regression_report: RegressionReport | None = None

    # Baseline gate decision. ``None`` means unknown (a report written before
    # gate decisions were recorded); ``GateResult.not_requested()`` means the
    # run had no gate. Rendered in every format and round-tripped via JSON.
    gate: GateResult | None = None

    # What was measured and which candidate was under test, copied from
    # ``TrialBatch.provenance``. ``None`` for legacy artifacts and hand-built
    # batches; rendered in Markdown and HTML and round-tripped via JSON.
    provenance: RunProvenance | None = None

    @property
    def excluded_trials(self) -> int:
        """Trials excluded from agent statistics."""
        return self.total_trials - self.gradable_trials

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_trials": self.total_trials,
            "total_tasks": self.total_tasks,
            "gradable_trials": self.gradable_trials,
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
            "metric_availability": {
                name: mv.to_dict() for name, mv in self.metric_availability.items()
            },
            "availability_recorded": self.availability_recorded,
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
        if self.gate is not None:
            result["gate"] = self.gate.to_dict()
        if self.provenance is not None:
            result["provenance"] = self.provenance.model_dump(mode="json")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReportData":
        """Rebuild from :meth:`to_dict` output.

        Reports written before availability was recorded load with
        ``availability_recorded=False``; their metric values are kept as
        recorded, eligibility is unknown, and ``gradable_trials`` falls back
        to ``total_trials`` (the legacy denominator). Reports written before
        gate decisions were recorded load with ``gate=None``; no decision is
        invented for them.

        Raises:
            ValueError: If ``data`` is not a TraceLens results document
                (not an object, or missing ``total_trials``, ``total_tasks``,
                or ``task_summaries``), or its ``provenance`` is invalid or
                uses a schema version this TraceLens does not know.
        """
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        missing = [
            key for key in ("total_trials", "total_tasks", "task_summaries")
            if key not in data
        ]
        if missing:
            raise ValueError("missing required keys: " + ", ".join(missing))
        if not isinstance(data["task_summaries"], list):
            raise ValueError("task_summaries must be a list")
        summaries = [TaskSummary(**s) for s in data["task_summaries"]]
        gate_data = data.get("gate")
        gate = GateResult.from_dict(gate_data) if isinstance(gate_data, dict) else None
        provenance: RunProvenance | None = None
        if data.get("provenance") is not None:
            try:
                provenance = RunProvenance.model_validate(data["provenance"])
            except ValueError as exc:  # pydantic ValidationError is a ValueError
                raise ValueError(f"invalid provenance: {exc}") from exc
        pass_at_k: dict[str, float | None] = dict(data.get("pass_at_k", {}))
        reliability: dict[str, float | None] = dict(data.get("reliability", {}))
        recorded = "metric_availability" in data
        if recorded:
            availability = {
                name: MetricValue.from_dict(entry)
                for name, entry in data["metric_availability"].items()
            }
        else:
            availability = {
                name: MetricValue.legacy(name, value)
                for name, value in {**pass_at_k, **reliability}.items()
            }
        total_trials = data.get("total_trials", 0)
        gradable = data.get("gradable_trials")
        return cls(
            total_trials=total_trials,
            total_tasks=data.get("total_tasks", 0),
            gradable_trials=total_trials if gradable is None else gradable,
            overall_pass_rate=data.get("overall_pass_rate", 0.0),
            overall_mean_score=data.get("overall_mean_score", 0.0),
            infra_error_count=data.get("infra_error_count", 0),
            infra_error_rate=data.get("infra_error_rate", 0.0),
            grader_error_count=data.get("grader_error_count", 0),
            grader_error_rate=data.get("grader_error_rate", 0.0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            task_summaries=summaries,
            pass_at_k=pass_at_k,
            reliability=reliability,
            metric_availability=availability,
            availability_recorded=recorded,
            gate=gate,
            provenance=provenance,
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
        """Build a ReportData from a TrialBatch.

        Only gradable trials (``Trial.is_gradable``) enter pass rates,
        scores, pass@k, and pass^k; harness failures are counted separately.
        Suite metrics that no task can support are reported as unavailable,
        never as zeros.
        """
        pass_results = batch.get_pass_results_by_task()
        # pass^k is a consecutive-window statistic: feed it run-indexed
        # sequences (None marks a missing or excluded run) so completion
        # order and gaps cannot change the reported reliability.
        pass_sequences = batch.get_pass_sequences_by_task()
        task_ids = sorted(pass_results.keys())
        task_hashes = (
            batch.provenance.measurement.task_hashes if batch.provenance is not None else {}
        )

        # Suite-level stats
        all_scores: list[float] = []
        task_summaries = []

        for task_id in task_ids:
            trials = batch.get_trials_for_task(task_id)
            gradable = [t for t in trials if t.is_gradable]
            passes = pass_results[task_id]

            scores = [
                t.aggregate_score
                for t in gradable
                if t.aggregate_score is not None
            ]

            pass_rate = sum(passes) / len(passes) if passes else 0.0
            mean_score = float(np.mean(scores)) if scores else 0.0
            std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

            task_pass_at_k = {
                name: mv.value
                for name, mv in self._pass_at_k.analyze_detailed(
                    {task_id: passes}
                ).items()
            }
            task_reliability = {
                name: mv.value
                for name, mv in self._consistency.analyze_detailed(
                    {task_id: pass_sequences[task_id]}
                ).items()
            }

            task_summaries.append(TaskSummary(
                task_id=task_id,
                num_trials=len(trials),
                pass_rate=pass_rate,
                mean_score=mean_score,
                std_score=std_score,
                pass_at_k=task_pass_at_k,
                reliability=task_reliability,
                gradable_trials=len(gradable),
                task_hash=task_hashes.get(task_id),
            ))

            all_scores.extend(scores)

        # Suite-level pass@k and reliability, with availability evidence
        pass_at_k_detail = self._pass_at_k.analyze_detailed(pass_results)
        reliability_detail = self._consistency.analyze_detailed(pass_sequences)

        report = ReportData(
            total_trials=batch.total_count,
            total_tasks=len(task_ids),
            gradable_trials=batch.gradable_count,
            overall_pass_rate=batch.pass_rate,
            overall_mean_score=float(np.mean(all_scores)) if all_scores else 0.0,
            infra_error_count=batch.infra_error_count,
            infra_error_rate=batch.infra_error_rate,
            grader_error_count=batch.grader_error_count,
            grader_error_rate=batch.grader_error_rate,
            total_input_tokens=batch.total_input_tokens,
            total_output_tokens=batch.total_output_tokens,
            task_summaries=task_summaries,
            pass_at_k={name: mv.value for name, mv in pass_at_k_detail.items()},
            reliability={name: mv.value for name, mv in reliability_detail.items()},
            metric_availability={**pass_at_k_detail, **reliability_detail},
            provenance=batch.provenance,
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
            f"- **Pass Rate**: {_format_pass_rate(report)}",
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
        if report.grader_error_count > 0:
            lines.append(
                f"- **Grader-Error Rate**: {report.grader_error_rate:.1%} "
                f"({report.grader_error_count} of {report.total_trials} trials)"
            )
            lines.append(
                "  - ⚠ A grader crashed on these trials. They are excluded from "
                "agent statistics; a spike here means the grading harness broke, "
                "not the agent."
            )
        if report.total_input_tokens or report.total_output_tokens:
            lines.append(
                f"- **Tokens**: {report.total_input_tokens:,} in / "
                f"{report.total_output_tokens:,} out"
            )
        lines.append("")

        lines.extend(_gate_section_md(report))
        lines.extend(_metric_section_md("Capability (pass@k)", report.pass_at_k, report))
        lines.extend(_metric_section_md("Reliability (pass^k)", report.reliability, report))

        # Per-task table
        if report.task_summaries:
            lines.append("## Per-Task Results")
            lines.append("")
            lines.append("| Task | Trials | Pass Rate | Mean Score |")
            lines.append("|------|--------|-----------|------------|")
            for s in report.task_summaries:
                lines.append(
                    f"| {s.task_id} | {_format_trial_count(s)} | "
                    f"{_format_task_pass_rate(s, report)} | {s.mean_score:.4f} |"
                )
            lines.append("")

        # Legacy hand-attached regression report (CLI runs use ``gate``)
        if (
            report.gate is None
            and report.regression_report
            and report.regression_report.has_regression
        ):
            lines.append("## Regression Alert")
            lines.append("")
            lines.append(report.regression_report.to_ci_output())
            lines.append("")

        if report.provenance is not None:
            lines.append("## Run Provenance")
            lines.append("")
            for label, value in _provenance_items(report.provenance):
                lines.append(f"- **{label}**: {value}")
            lines.append("")
            lines.append(
                "These are declared identities and content hashes: evidence for "
                "attributing a change, not proof of identical execution."
            )
            lines.append("")

        return "\n".join(lines)

    def render_ci_summary(self, report: ReportData) -> str:
        """Render a compact CI-friendly summary.

        Unavailable metrics render as ``n/a`` so a gate cannot mistake
        "not measured" for a measured zero.
        """
        if report.availability_recorded and report.gradable_trials == 0:
            pass_rate_text = "n/a"
        else:
            pass_rate_text = f"{report.overall_pass_rate:.1%}"
        lines = [
            f"TraceLens: {report.total_tasks} tasks, "
            f"{report.total_trials} trials, "
            f"pass_rate={pass_rate_text}, "
            f"mean_score={report.overall_mean_score:.4f}",
        ]

        for key, val in sorted(report.pass_at_k.items()):
            lines[0] += f", {key}=" + ("n/a" if val is None else f"{val:.4f}")

        if report.infra_error_count > 0:
            lines[0] += f", infra_errors={report.infra_error_rate:.1%}"
        if report.grader_error_count > 0:
            lines[0] += f", grader_errors={report.grader_error_rate:.1%}"

        if report.gate is not None and report.gate.requested:
            # Blocking tasks print the detector's own text, then the one-line
            # gate summary -- the same lines the CLI used to assemble itself.
            for task in report.gate.tasks:
                if task.outcome is TaskGateOutcome.CHECKED and task.blocking:
                    lines.append(task.regression_report().to_ci_output())
            lines.append(report.gate.summary_line())
        elif report.regression_report and report.regression_report.has_regression:
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
        if report.availability_recorded and report.gradable_trials == 0:
            pass_rate_text = "N/A"
            pass_rate_color = "#6b7280"
        else:
            pass_rate_text = f"{pass_rate_pct:.1f}%"

        # --- Summary cards ---
        cards_html = (
            _html_card("Tasks", str(report.total_tasks), "#3b82f6")
            + _html_card("Trials", str(report.total_trials), "#6366f1")
            + _html_card("Pass Rate", pass_rate_text, pass_rate_color)
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
        if report.grader_error_count > 0:
            cards_html += _html_card(
                "Grader Errors",
                f"{report.grader_error_rate * 100:.1f}%",
                "#ef4444" if report.grader_error_rate >= 0.05 else "#f97316",
            )
        gate_html = _gate_section_html(report)
        provenance_html = _provenance_section_html(report)

        # --- Capability / Reliability bar charts (available metrics only;
        # unavailable ones are listed with their reason, never drawn as 0) ---
        capability_svg = _metric_section_html(report, report.pass_at_k)
        reliability_svg = _metric_section_html(report, report.reliability)

        # --- Per-task table ---
        task_rows = ""
        for s in report.task_summaries:
            pr_color = _pass_rate_color(s.pass_rate)
            task_rows += (
                f'<tr><td>{escape(s.task_id)}</td>'
                f"<td>{escape(_format_trial_count(s))}</td>"
                f'<td style="color:{pr_color};font-weight:600">'
                f"{escape(_format_task_pass_rate(s, report))}</td>"
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

        # --- Legacy regression alert (CLI runs render ``gate`` instead) ---
        regression_html = ""
        if (
            report.gate is None
            and report.regression_report
            and report.regression_report.has_regression
        ):
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
  .na {{ color: #64748b; font-size: 0.9em; margin: 6px 0; }}
  .badge {{ color: #fff; padding: 2px 10px; border-radius: 12px; font-size: 0.75em;
            margin-left: 8px; }}
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

{gate_html}

{"<section><h2>Capability (pass@k)</h2>" + capability_svg + "</section>" if capability_svg else ""}
{"<section><h2>Reliability (pass^k)</h2>" + reliability_svg + "</section>" if reliability_svg else ""}

{"<section><h2>Per-Task Results</h2><table><thead><tr><th>Task</th><th>Trials</th><th>Pass Rate</th><th>Mean Score</th><th>Std Score</th></tr></thead><tbody>" + task_rows + "</tbody></table></section>" if task_rows else ""}

{"<section><h2>Pass Rate Distribution</h2>" + pass_rate_chart + "</section>" if pass_rate_chart else ""}

{"<section><h2>Score Distribution</h2>" + score_histogram + "</section>" if score_histogram else ""}

{regression_html}

{provenance_html}

<div class="subtitle" style="margin-top:24px;text-align:center">
  TraceLens v{__version__} &middot; {escape(timestamp)}
</div>
</body>
</html>"""


_GATE_COLORS = {
    GateStatus.PASSED: "#22c55e",
    GateStatus.BLOCKED: "#ef4444",
    GateStatus.UNEVALUABLE: "#f97316",
    GateStatus.NOT_REQUESTED: "#6b7280",
}


def _gate_status_text(gate: GateResult) -> str:
    if gate.status is GateStatus.NOT_REQUESTED:
        return "not requested (run without `--baseline-check`)"
    return f"{gate.status.value.upper()} (exit code {gate.exit_code})"


def _gate_policy_text(gate: GateResult) -> str:
    threshold = gate.threshold.value if gate.threshold else "moderate"
    band = f"{gate.noise_band}" if gate.noise_band is not None else "default"
    required = "yes" if gate.require_baselines else "no"
    return (
        f"block at `{threshold}` or worse; noise band {band}; "
        f"require baselines: {required}"
    )


def _gate_task_counts(gate: GateResult) -> str:
    text = (
        f"{gate.checked} checked, {gate.skipped_no_baseline} skipped (no baseline), "
        f"{gate.skipped_no_gradable} skipped (no gradable trials), "
        f"{gate.skipped_no_comparable_metrics} skipped (no comparable metrics)"
    )
    if gate.skipped_task_content_changed:
        text += f", {gate.skipped_task_content_changed} skipped (task content changed)"
    return text


def _provenance_items(provenance: RunProvenance) -> list[tuple[str, str]]:
    """``(label, value)`` pairs from :meth:`RunProvenance.summary_lines`."""
    items = []
    for line in provenance.summary_lines():
        label, _, value = line.partition(": ")
        items.append((label, value))
    return items


def _provenance_section_html(report: ReportData) -> str:
    if report.provenance is None:
        return ""
    rows = "".join(
        f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"
        for label, value in _provenance_items(report.provenance)
    )
    return (
        "<section><h2>Run Provenance</h2><table><tbody>" + rows + "</tbody></table>"
        '<p class="na">Declared identities and content hashes: evidence for '
        "attributing a change, not proof of identical execution.</p></section>"
    )


def _regression_notes(task: Any, regression: Any) -> str:
    notes: list[str] = []
    if regression.within_noise_band:
        notes.append("within infra-noise band; not blocking")
    elif task.blocking:
        notes.append("blocking")
    if regression.insufficient_data:
        notes.append("insufficient samples; severity from thresholds")
    if task.infra_config_mismatch:
        notes.append("infra config differs from baseline")
    return "; ".join(notes)


def _gate_rows(gate: GateResult) -> list[tuple[str, str, str, str, str, str, str]]:
    rows = []
    for task in gate.tasks:
        for regression in task.regressions:
            rows.append((
                task.task_id,
                regression.metric_name,
                f"{regression.baseline_mean:.4f}",
                f"{regression.current_mean:.4f}",
                f"{regression.delta_percent:+.1f}%",
                regression.severity.value,
                _regression_notes(task, regression),
            ))
    return rows


def _gate_skipped_lines(gate: GateResult) -> list[str]:
    return [
        f"{task.task_id} ({task.reason})"
        for task in gate.tasks
        if task.outcome is not TaskGateOutcome.CHECKED
    ]


def _gate_section_md(report: ReportData) -> list[str]:
    gate = report.gate
    if gate is None:
        return []
    lines = ["## Baseline Gate", "", f"- **Status**: {_gate_status_text(gate)}"]
    if gate.requested:
        lines.append(f"- **Policy**: {_gate_policy_text(gate)}")
        lines.append(f"- **Tasks**: {_gate_task_counts(gate)}")
        lines.append(f"- **Blocking regressions**: {gate.blocking_regressions}")
        for reason in gate.reasons:
            lines.append(f"- **Why**: {reason}")
        for warning in gate.warnings:
            lines.append(f"- **Warning**: {warning}")
        rows = _gate_rows(gate)
        if rows:
            lines.append("")
            lines.append(
                "| Task | Metric | Baseline | Current | Change | Severity | Notes |"
            )
            lines.append(
                "|------|--------|----------|---------|--------|----------|-------|"
            )
            for row in rows:
                lines.append("| " + " | ".join(row) + " |")
        skipped = _gate_skipped_lines(gate)
        if skipped:
            lines.append("")
            lines.append("Skipped tasks: " + "; ".join(skipped))
    lines.append("")
    return lines


def _gate_section_html(report: ReportData) -> str:
    gate = report.gate
    if gate is None:
        return ""
    color = _GATE_COLORS[gate.status]
    badge = (
        f'<span class="badge" style="background:{color}">'
        f"{escape(gate.status.value.upper())}</span>"
    )
    body = f"<p><strong>Status</strong>: {escape(_gate_status_text(gate))}</p>"
    if gate.requested:
        body += f"<p><strong>Policy</strong>: {escape(_gate_policy_text(gate))}</p>"
        body += f"<p><strong>Tasks</strong>: {escape(_gate_task_counts(gate))}</p>"
        body += (
            f"<p><strong>Blocking regressions</strong>: {gate.blocking_regressions}</p>"
        )
        for reason in gate.reasons:
            body += f"<p><strong>Why</strong>: {escape(reason)}</p>"
        for warning in gate.warnings:
            body += f'<p class="na"><strong>Warning</strong>: {escape(warning)}</p>'
        rows = _gate_rows(gate)
        if rows:
            body += (
                "<table><thead><tr><th>Task</th><th>Metric</th><th>Baseline</th>"
                "<th>Current</th><th>Change</th><th>Severity</th><th>Notes</th>"
                "</tr></thead><tbody>"
            )
            for row in rows:
                body += "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>"
            body += "</tbody></table>"
        skipped = _gate_skipped_lines(gate)
        if skipped:
            body += f'<p class="na">Skipped tasks: {escape("; ".join(skipped))}</p>'
    return f"<section><h2>Baseline Gate{badge}</h2>{body}</section>"


def _format_pass_rate(report: ReportData) -> str:
    """Suite pass rate with its denominator, or N/A when nothing is gradable."""
    if not report.availability_recorded:
        return f"{report.overall_pass_rate:.1%}"
    if report.gradable_trials == 0:
        return "N/A (no gradable trials)"
    text = f"{report.overall_pass_rate:.1%}"
    if report.excluded_trials > 0:
        text += (
            f" (over {report.gradable_trials} gradable trials; "
            f"{report.excluded_trials} excluded as harness failures or never run)"
        )
    return text


def _format_task_pass_rate(summary: TaskSummary, report: ReportData) -> str:
    if report.availability_recorded and summary.gradable_trials == 0:
        return "N/A"
    return f"{summary.pass_rate:.1%}"


def _format_trial_count(summary: TaskSummary) -> str:
    if summary.gradable_trials is None or summary.gradable_trials == summary.num_trials:
        return str(summary.num_trials)
    return f"{summary.num_trials} ({summary.gradable_trials} gradable)"


def _describe_metric(name: str, value: float | None, report: ReportData) -> str:
    mv = report.metric_availability.get(name)
    if mv is not None:
        return mv.describe()
    return "N/A" if value is None else f"{value:.4f}"


def _runs_hint(names: list[str], report: ReportData) -> int | None:
    """Largest ``k`` among unavailable metrics that more runs would unlock."""
    needed = [
        mv.required_runs
        for name in names
        if (mv := report.metric_availability.get(name)) is not None
        and not mv.available
        and mv.required_runs is not None
        and (mv.total_tasks or 0) > 0
    ]
    return max(needed) if needed else None


def _metric_section_md(
    title: str, metrics: dict[str, float | None], report: ReportData
) -> list[str]:
    if not metrics:
        return []
    lines = [f"## {title}", ""]
    for key in sorted(metrics):
        lines.append(f"- **{key}**: {_describe_metric(key, metrics[key], report)}")
    needed = _runs_hint(sorted(metrics), report)
    if needed is not None:
        lines.append(
            f"  - Unavailable metrics need more gradable runs per task; "
            f"rerun with `--num-runs {needed}`."
        )
    if not report.availability_recorded:
        lines.append(
            "  - Metric availability was not recorded in this report; a zero "
            "may be an unavailable metric rather than a measured zero."
        )
    lines.append("")
    return lines


def _metric_section_html(report: ReportData, metrics: dict[str, float | None]) -> str:
    if not metrics:
        return ""
    labels = [k for k in sorted(metrics) if metrics[k] is not None]
    values: list[float] = []
    for label in labels:
        value = metrics[label]
        if value is not None:
            values.append(value)
    html = _svg_bar_chart(labels, values, 1.0) if labels else ""
    for key in sorted(metrics):
        if metrics[key] is None:
            html += (
                f'<p class="na"><strong>{escape(key)}</strong>: '
                f"{escape(_describe_metric(key, None, report))}</p>"
            )
    needed = _runs_hint(sorted(metrics), report)
    if needed is not None:
        html += (
            f'<p class="na">Unavailable metrics need more gradable runs per task; '
            f"rerun with <code>--num-runs {needed}</code>.</p>"
        )
    if not report.availability_recorded:
        html += (
            '<p class="na">Metric availability was not recorded in this report; '
            "a zero may be an unavailable metric rather than a measured zero.</p>"
        )
    return html


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
