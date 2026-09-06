"""Baseline gate decisions.

``tracelens run --baseline-check`` compares each task's current trials with
its stored baseline and decides whether the run may proceed. That decision
used to exist only as CLI text and an exit code: the JSON, Markdown, and
HTML reports were written before it was made, and re-rendering a saved
report lost it. :class:`GateResult` is the single record of the decision.
The CLI derives its exit code from it, every report format renders it, and
``ReportData.to_dict`` / ``from_dict`` round-trip it.

Statuses:

- ``not_requested`` -- the run had no ``--baseline-check`` (exit 0).
- ``passed`` -- at least one task was compared and nothing blocked (exit 0).
- ``blocked`` -- a regression at or above the threshold, or
  ``--require-baselines`` with a task that has no baseline (exit 1).
- ``unevaluable`` -- no task could be compared, or a baseline-backed task
  had no gradable trials or no comparable metric. Missing evidence never
  authorizes a passing gate (exit 2), and it takes precedence over
  ``blocked``.

Trial validity follows ``docs/statistical-contract.md``: only gradable
trials (``Trial.is_gradable``) enter a comparison; harness failures are
counted as excluded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from tracelens.baselines.comparison import (
    DEFAULT_NOISE_BAND_ABSOLUTE,
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
)
from tracelens.baselines.manager import BaselineManager
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.provenance import short_hash
from tracelens.core.trial import Trial, TrialBatch


class GateStatus(StrEnum):
    """Outcome of a baseline gate."""

    NOT_REQUESTED = "not_requested"
    PASSED = "passed"
    BLOCKED = "blocked"
    UNEVALUABLE = "unevaluable"


class TaskGateOutcome(StrEnum):
    """What happened to one task inside the gate."""

    CHECKED = "checked"
    NO_BASELINE = "no_baseline"
    NO_GRADABLE_TRIALS = "no_gradable_trials"
    NO_COMPARABLE_METRICS = "no_comparable_metrics"
    TASK_CONTENT_CHANGED = "task_content_changed"


EXIT_CODES: dict[GateStatus, int] = {
    GateStatus.NOT_REQUESTED: 0,
    GateStatus.PASSED: 0,
    GateStatus.BLOCKED: 1,
    GateStatus.UNEVALUABLE: 2,
}

# The task-level metrics the CLI compares against stored baselines.
CLI_METRICS = ("pass_rate", "mean_score")


def per_trial_results(trials: Sequence[Trial]) -> list[dict[str, float]]:
    """One metric sample per gradable trial for regression detection.

    ``RegressionDetector.compare()`` runs a t-test over the sample
    distribution, so it needs per-trial values; a pre-aggregated single dict
    would collapse it to a one-sample z-test. The sample mean of the
    per-trial ``pass_rate`` indicators equals the task's pass rate, so
    baseline metric names stay unchanged.

    Only gradable trials contribute (statistical contract): harness
    failures and trials that never ran are excluded and surfaced separately.
    ``TIMEOUT`` stays included as a failure -- blowing the time budget is an
    agent-quality signal.
    """
    results: list[dict[str, float]] = []
    for trial in trials:
        if not trial.is_gradable:
            continue
        results.append({
            "pass_rate": 1.0 if trial.passed else 0.0,
            "mean_score": (
                trial.aggregate_score if trial.aggregate_score is not None else 0.0
            ),
        })
    return results


def spec_from_trials(
    trials: Sequence[Trial],
) -> tuple[DecisionSpec | None, str | None]:
    """Recover the run's DecisionSpec from adapter-stamped transcripts.

    Returns the most recent spec and, when the trials carry more than one
    distinct spec (a checkpoint resume with a changed configuration), a
    warning text for the caller to surface. The most recent spec wins
    because resumed trials are loaded before new ones run.
    """
    specs = [
        trial.transcript.decision_spec
        for trial in trials
        if trial.transcript is not None and trial.transcript.decision_spec is not None
    ]
    if not specs:
        return None, None
    warning = None
    if len({spec.fingerprint for spec in specs}) > 1:
        warning = (
            "mixed decision specs found across trials (checkpoint resume with "
            "a changed config?); using the most recent -- pass --decision-spec "
            "to be explicit"
        )
    return specs[-1], warning


def _diff_to_json(diff: dict[str, tuple[Any, Any]]) -> dict[str, list[Any]]:
    return {key: [baseline, current] for key, (baseline, current) in diff.items()}


def _diff_from_json(data: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    return {key: (pair[0], pair[1]) for key, pair in data.items()}


@dataclass
class TaskGateResult:
    """The gate's view of one task."""

    task_id: str
    outcome: TaskGateOutcome
    reason: str | None = None
    compared_trials: int = 0
    excluded_trials: int = 0
    available_metrics: list[str] = field(default_factory=list)
    blocking: bool = False
    has_regression: bool = False
    overall_severity: RegressionSeverity = RegressionSeverity.NONE
    infra_config_mismatch: bool = False
    infra_config_diff: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    regressions: list[MetricRegression] = field(default_factory=list)
    improvements: list[MetricRegression] = field(default_factory=list)

    def regression_report(self) -> RegressionReport:
        """Rebuild the detector's report so all text comes from one formatter."""
        return RegressionReport(
            has_regression=self.has_regression,
            overall_severity=self.overall_severity,
            regressions=list(self.regressions),
            improvements=list(self.improvements),
            infra_config_mismatch=self.infra_config_mismatch,
            infra_config_diff=dict(self.infra_config_diff),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "compared_trials": self.compared_trials,
            "excluded_trials": self.excluded_trials,
            "available_metrics": list(self.available_metrics),
            "blocking": self.blocking,
            "has_regression": self.has_regression,
            "overall_severity": self.overall_severity.value,
            "infra_config_mismatch": self.infra_config_mismatch,
            "infra_config_diff": _diff_to_json(self.infra_config_diff),
            "regressions": [r.model_dump(mode="json") for r in self.regressions],
            "improvements": [r.model_dump(mode="json") for r in self.improvements],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGateResult:
        return cls(
            task_id=str(data["task_id"]),
            outcome=TaskGateOutcome(data["outcome"]),
            reason=data.get("reason"),
            compared_trials=int(data.get("compared_trials", 0)),
            excluded_trials=int(data.get("excluded_trials", 0)),
            available_metrics=list(data.get("available_metrics", [])),
            blocking=bool(data.get("blocking", False)),
            has_regression=bool(data.get("has_regression", False)),
            overall_severity=RegressionSeverity(data.get("overall_severity", "none")),
            infra_config_mismatch=bool(data.get("infra_config_mismatch", False)),
            infra_config_diff=_diff_from_json(data.get("infra_config_diff", {})),
            regressions=[
                MetricRegression.model_validate(r) for r in data.get("regressions", [])
            ],
            improvements=[
                MetricRegression.model_validate(r) for r in data.get("improvements", [])
            ],
        )


@dataclass
class GateResult:
    """The baseline gate decision for one run."""

    status: GateStatus
    exit_code: int
    threshold: RegressionSeverity | None = None
    noise_band: float | None = None
    require_baselines: bool = False
    checked: int = 0
    skipped_no_baseline: int = 0
    skipped_no_gradable: int = 0
    skipped_no_comparable_metrics: int = 0
    skipped_task_content_changed: int = 0
    blocking_regressions: int = 0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tasks: list[TaskGateResult] = field(default_factory=list)

    @classmethod
    def not_requested(cls) -> GateResult:
        """The record for a run that had no ``--baseline-check``."""
        return cls(status=GateStatus.NOT_REQUESTED, exit_code=0)

    @property
    def requested(self) -> bool:
        return self.status is not GateStatus.NOT_REQUESTED

    def tasks_with(self, outcome: TaskGateOutcome) -> list[TaskGateResult]:
        return [t for t in self.tasks if t.outcome is outcome]

    def summary_line(self) -> str:
        """The one-line gate summary printed by the CLI and in every format."""
        parts = [
            f"{self.checked} checked",
            f"{self.skipped_no_baseline} skipped (no baseline)",
        ]
        if self.skipped_no_gradable:
            parts.append(f"{self.skipped_no_gradable} skipped (no gradable trials)")
        if self.skipped_no_comparable_metrics:
            parts.append(
                f"{self.skipped_no_comparable_metrics} skipped (no comparable metrics)"
            )
        if self.skipped_task_content_changed:
            parts.append(
                f"{self.skipped_task_content_changed} skipped (task content changed)"
            )
        parts.append(f"{self.blocking_regressions} blocking regression(s)")
        if self.status is GateStatus.UNEVALUABLE:
            parts.append("UNEVALUABLE")
        return f"[tracelens] Baseline check: {', '.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "threshold": self.threshold.value if self.threshold else None,
            "noise_band": self.noise_band,
            "require_baselines": self.require_baselines,
            "checked": self.checked,
            "skipped_no_baseline": self.skipped_no_baseline,
            "skipped_no_gradable": self.skipped_no_gradable,
            "skipped_no_comparable_metrics": self.skipped_no_comparable_metrics,
            "skipped_task_content_changed": self.skipped_task_content_changed,
            "blocking_regressions": self.blocking_regressions,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateResult:
        threshold = data.get("threshold")
        return cls(
            status=GateStatus(data["status"]),
            exit_code=int(data.get("exit_code", EXIT_CODES[GateStatus(data["status"])])),
            threshold=RegressionSeverity(threshold) if threshold else None,
            noise_band=data.get("noise_band"),
            require_baselines=bool(data.get("require_baselines", False)),
            checked=int(data.get("checked", 0)),
            skipped_no_baseline=int(data.get("skipped_no_baseline", 0)),
            skipped_no_gradable=int(data.get("skipped_no_gradable", 0)),
            skipped_no_comparable_metrics=int(data.get("skipped_no_comparable_metrics", 0)),
            skipped_task_content_changed=int(data.get("skipped_task_content_changed", 0)),
            blocking_regressions=int(data.get("blocking_regressions", 0)),
            reasons=list(data.get("reasons", [])),
            warnings=list(data.get("warnings", [])),
            tasks=[TaskGateResult.from_dict(t) for t in data.get("tasks", [])],
        )


def evaluate_gate(
    batch: TrialBatch,
    baseline_manager: BaselineManager,
    *,
    threshold: RegressionSeverity = RegressionSeverity.MODERATE,
    noise_band: float = DEFAULT_NOISE_BAND_ABSOLUTE,
    require_baselines: bool = False,
    decision_spec: DecisionSpec | None = None,
    task_ids: Sequence[str] | None = None,
    task_hashes: Mapping[str, str] | None = None,
) -> GateResult:
    """Compare a run against stored baselines and decide the gate.

    Args:
        batch: The run's trials.
        baseline_manager: Loaded baselines to compare against.
        threshold: Minimum regression severity that blocks.
        noise_band: Absolute delta treated as infra noise when both sides
            carry a ``DecisionSpec`` whose infra configs differ.
        require_baselines: Block when any task has no stored baseline.
        decision_spec: The run's spec; when ``None`` it is recovered from
            adapter-stamped transcripts per task.
        task_ids: Tasks to consider, in order. Defaults to every task in the
            batch, sorted.
        task_hashes: Content hash per task for the current run. Defaults to
            the batch's recorded provenance. A baseline that stores a
            ``task_hash`` is compared only when it matches; a task whose
            content changed since its baseline is never silently compared
            by id, it makes the gate unevaluable until re-baselined.

    Returns:
        A :class:`GateResult` with one :class:`TaskGateResult` per task.
    """
    detector = RegressionDetector(noise_band_absolute=noise_band)
    trials_by_task: dict[str, list[Trial]] = {}
    for trial in batch.trials:
        trials_by_task.setdefault(trial.task_id, []).append(trial)
    ordered = list(task_ids) if task_ids is not None else sorted(trials_by_task)
    if task_hashes is None:
        task_hashes = (
            batch.provenance.measurement.task_hashes if batch.provenance is not None else {}
        )
    unhashed_baselines: list[str] = []

    tasks: list[TaskGateResult] = []
    warnings: list[str] = []
    for task_id in ordered:
        task_trials = trials_by_task.get(task_id, [])
        baseline = baseline_manager.get_baseline(task_id)
        current_results = per_trial_results(task_trials)
        excluded = len(task_trials) - len(current_results)
        if baseline is None:
            tasks.append(TaskGateResult(
                task_id=task_id,
                outcome=TaskGateOutcome.NO_BASELINE,
                reason="no baseline stored for this task",
                compared_trials=0,
                excluded_trials=excluded,
            ))
            continue
        current_hash = task_hashes.get(task_id)
        if baseline.task_hash and current_hash and baseline.task_hash != current_hash:
            tasks.append(TaskGateResult(
                task_id=task_id,
                outcome=TaskGateOutcome.TASK_CONTENT_CHANGED,
                reason=(
                    "task content changed since the baseline was stored "
                    f"({short_hash(baseline.task_hash)} -> {short_hash(current_hash)}); "
                    "re-store the baseline for this task"
                ),
                excluded_trials=excluded,
            ))
            continue
        if current_hash and not baseline.task_hash:
            unhashed_baselines.append(task_id)
        if not current_results:
            tasks.append(TaskGateResult(
                task_id=task_id,
                outcome=TaskGateOutcome.NO_GRADABLE_TRIALS,
                reason="no gradable trials (all infra/grader failures)",
                excluded_trials=excluded,
            ))
            continue
        current_metrics = sorted({name for result in current_results for name in result})
        if not baseline.metrics.keys() & set(current_metrics):
            tasks.append(TaskGateResult(
                task_id=task_id,
                outcome=TaskGateOutcome.NO_COMPARABLE_METRICS,
                reason=(
                    "baseline shares no metric with the CLI metrics "
                    f"({', '.join(current_metrics)})"
                ),
                compared_trials=len(current_results),
                excluded_trials=excluded,
                available_metrics=current_metrics,
            ))
            continue
        current_spec = decision_spec
        if current_spec is None:
            current_spec, warning = spec_from_trials(task_trials)
            if warning and warning not in warnings:
                warnings.append(warning)
        report = detector.compare_with_specs(
            baseline,
            current_results,
            baseline_spec=baseline.decision_spec,
            current_spec=current_spec,
        )
        tasks.append(TaskGateResult(
            task_id=task_id,
            outcome=TaskGateOutcome.CHECKED,
            compared_trials=len(current_results),
            excluded_trials=excluded,
            available_metrics=current_metrics,
            blocking=report.should_block_ci(threshold),
            has_regression=report.has_regression,
            overall_severity=report.overall_severity,
            infra_config_mismatch=report.infra_config_mismatch,
            infra_config_diff=dict(report.infra_config_diff),
            regressions=list(report.regressions),
            improvements=list(report.improvements),
        ))

    checked = [t for t in tasks if t.outcome is TaskGateOutcome.CHECKED]
    no_baseline = [t for t in tasks if t.outcome is TaskGateOutcome.NO_BASELINE]
    no_gradable = [t for t in tasks if t.outcome is TaskGateOutcome.NO_GRADABLE_TRIALS]
    no_comparable = [t for t in tasks if t.outcome is TaskGateOutcome.NO_COMPARABLE_METRICS]
    content_changed = [
        t for t in tasks if t.outcome is TaskGateOutcome.TASK_CONTENT_CHANGED
    ]
    blocking = [t for t in checked if t.blocking]
    if unhashed_baselines:
        warnings.append(
            f"{len(unhashed_baselines)} baseline(s) carry no task_hash, so a change to "
            "their task content cannot be detected: " + ", ".join(unhashed_baselines)
            + "; re-store them from a results file that records provenance"
        )

    reasons: list[str] = []
    if not checked or no_gradable or no_comparable or content_changed:
        status = GateStatus.UNEVALUABLE
        if not checked:
            reasons.append("no task could be compared against a baseline")
        if content_changed:
            reasons.append(
                f"{len(content_changed)} task(s) whose content changed since their "
                "baseline was stored: " + ", ".join(t.task_id for t in content_changed)
            )
        if no_gradable:
            reasons.append(
                f"{len(no_gradable)} task(s) with no gradable trials: "
                + ", ".join(t.task_id for t in no_gradable)
            )
        if no_comparable:
            reasons.append(
                f"{len(no_comparable)} task(s) with no comparable metrics: "
                + ", ".join(t.task_id for t in no_comparable)
            )
    else:
        status = GateStatus.PASSED
        if require_baselines and no_baseline:
            status = GateStatus.BLOCKED
            reasons.append(
                f"--require-baselines set but {len(no_baseline)} task(s) have no "
                "baseline: " + ", ".join(t.task_id for t in no_baseline)
            )
        if blocking:
            status = GateStatus.BLOCKED
            reasons.append(
                f"{len(blocking)} blocking regression(s) at threshold "
                f"'{threshold.value}': "
                + ", ".join(
                    f"{t.task_id} ({t.overall_severity.value})" for t in blocking
                )
            )
        if status is GateStatus.PASSED:
            reasons.append(
                f"{len(checked)} task(s) compared; no regression at or above "
                f"'{threshold.value}'"
            )

    return GateResult(
        status=status,
        exit_code=EXIT_CODES[status],
        threshold=threshold,
        noise_band=noise_band,
        require_baselines=require_baselines,
        checked=len(checked),
        skipped_no_baseline=len(no_baseline),
        skipped_no_gradable=len(no_gradable),
        skipped_no_comparable_metrics=len(no_comparable),
        skipped_task_content_changed=len(content_changed),
        blocking_regressions=len(blocking),
        reasons=reasons,
        warnings=warnings,
        tasks=tasks,
    )
