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
- ``unevaluable`` -- no task could be compared, a baseline-backed task had
  no gradable trials or no comparable metric, or no test in the run could
  have rejected at these sample sizes. Missing evidence never authorizes a
  passing gate (exit 2), and it takes precedence over ``blocked``.

The decision procedure (``docs/statistical-contract.md``, "Baseline
regression detection") has two criteria, each held to the significance
level ``alpha``:

1. **Per task.** :class:`~tracelens.baselines.comparison.RegressionDetector`
   tests every metric; the per-task p-values of each metric are
   Holm-adjusted across the checked tasks (``multiplicity="holm"``) so that
   the chance of any false block among them is at most ``alpha``. A task
   blocks when a significant regression reaches the severity threshold.
2. **Suite.** The mean of the per-task differences (current minus baseline
   mean, one per checked task) with the contract's task bootstrap and
   sign-flip test. It catches a broad regression that no single task can
   show, and blocks when it is significant and at or above the threshold.

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
    DEFAULT_SIGNIFICANCE_LEVEL,
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
    holm_adjusted,
    relative_change,
    severity_at_least,
    severity_for,
)
from tracelens.baselines.manager import BaselineManager
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.provenance import short_hash
from tracelens.core.trial import Trial, TrialBatch
from tracelens.statistics.run_comparison import paired_task_effect


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

# Multiplicity control across the checked tasks.
MULTIPLICITY_CHOICES = ("holm", "none")

# Suite-level bootstrap settings (the contract's run-versus-run defaults).
_SUITE_CONFIDENCE = 0.95
_SUITE_BOOTSTRAP = 10000


def per_trial_results(trials: Sequence[Trial]) -> list[dict[str, float]]:
    """One metric sample per gradable trial for regression detection.

    ``RegressionDetector.compare()`` tests the sample distribution, so it
    needs per-trial values; a pre-aggregated single dict would collapse it
    to a single observation. The sample mean of the per-trial ``pass_rate``
    indicators equals the task's pass rate, so baseline metric names stay
    unchanged.

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
    # The metrics the baseline and the run share: what was actually tested.
    compared_metrics: list[str] = field(default_factory=list)
    blocking: bool = False
    has_regression: bool = False
    overall_severity: RegressionSeverity = RegressionSeverity.NONE
    infra_config_mismatch: bool = False
    infra_config_diff: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    regressions: list[MetricRegression] = field(default_factory=list)
    improvements: list[MetricRegression] = field(default_factory=list)
    # Whether a total failure of this task could have blocked at these
    # sample sizes, and if not, the trials per task that would let it.
    # ``None`` for tasks that were not checked.
    detectable: bool | None = None
    trials_needed: int | None = None

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

    @property
    def underpowered_regressions(self) -> list[MetricRegression]:
        """Observed drops the evidence could not confirm."""
        return [r for r in self.regressions if not r.is_significant]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
            "compared_trials": self.compared_trials,
            "excluded_trials": self.excluded_trials,
            "available_metrics": list(self.available_metrics),
            "compared_metrics": list(self.compared_metrics),
            "blocking": self.blocking,
            "has_regression": self.has_regression,
            "overall_severity": self.overall_severity.value,
            "infra_config_mismatch": self.infra_config_mismatch,
            "infra_config_diff": _diff_to_json(self.infra_config_diff),
            "regressions": [r.model_dump(mode="json") for r in self.regressions],
            "improvements": [r.model_dump(mode="json") for r in self.improvements],
            "detectable": self.detectable,
            "trials_needed": self.trials_needed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskGateResult:
        detectable = data.get("detectable")
        trials_needed = data.get("trials_needed")
        return cls(
            task_id=str(data["task_id"]),
            outcome=TaskGateOutcome(data["outcome"]),
            reason=data.get("reason"),
            compared_trials=int(data.get("compared_trials", 0)),
            excluded_trials=int(data.get("excluded_trials", 0)),
            available_metrics=list(data.get("available_metrics", [])),
            compared_metrics=list(data.get("compared_metrics", [])),
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
            detectable=None if detectable is None else bool(detectable),
            trials_needed=None if trials_needed is None else int(trials_needed),
        )


@dataclass
class SuiteGateResult:
    """The suite-level criterion for one metric.

    ``delta`` is the mean over checked tasks of (current mean - baseline
    mean); ``p_value`` is one-sided in the regression direction from the
    contract's sign-flip test, with a task-bootstrap interval on the mean.
    """

    metric_name: str
    tasks: int
    baseline_mean: float
    current_mean: float
    delta: float
    delta_percent: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    p_value_exact: bool = False
    confidence: float = _SUITE_CONFIDENCE
    n_bootstrap: int = _SUITE_BOOTSTRAP
    seed: int | None = 0
    severity: RegressionSeverity = RegressionSeverity.NONE
    is_regression: bool = False
    is_significant: bool = False
    within_noise_band: bool = False
    blocking: bool = False

    def describe(self) -> str:
        """One line for the reports."""
        text = (
            f"{self.metric_name}: {self.baseline_mean:.4f} -> {self.current_mean:.4f} "
            f"({self.delta_percent:+.1f}%) over {self.tasks} task(s)"
        )
        if self.ci_lower is not None and self.ci_upper is not None:
            text += (
                f"; {self.confidence:.0%} CI of the mean difference "
                f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
            )
        if self.p_value is not None:
            text += f"; p={self.p_value:.4f}"
        if not self.is_regression:
            text += "; no drop"
        elif self.blocking:
            text += f"; significant, {self.severity.value}: blocking"
        elif self.within_noise_band:
            text += "; within infra-noise band; not blocking"
        elif self.is_significant:
            text += f"; significant, {self.severity.value}: below the threshold"
        else:
            text += "; not significant"
        return text

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "tasks": self.tasks,
            "baseline_mean": self.baseline_mean,
            "current_mean": self.current_mean,
            "delta": self.delta,
            "delta_percent": self.delta_percent,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
            "p_value_exact": self.p_value_exact,
            "confidence": self.confidence,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
            "severity": self.severity.value,
            "is_regression": self.is_regression,
            "is_significant": self.is_significant,
            "within_noise_band": self.within_noise_band,
            "blocking": self.blocking,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuiteGateResult:
        return cls(
            metric_name=str(data["metric_name"]),
            tasks=int(data.get("tasks", 0)),
            baseline_mean=float(data.get("baseline_mean", 0.0)),
            current_mean=float(data.get("current_mean", 0.0)),
            delta=float(data.get("delta", 0.0)),
            delta_percent=float(data.get("delta_percent", 0.0)),
            ci_lower=data.get("ci_lower"),
            ci_upper=data.get("ci_upper"),
            p_value=data.get("p_value"),
            p_value_exact=bool(data.get("p_value_exact", False)),
            confidence=float(data.get("confidence", _SUITE_CONFIDENCE)),
            n_bootstrap=int(data.get("n_bootstrap", _SUITE_BOOTSTRAP)),
            seed=data.get("seed"),
            severity=RegressionSeverity(data.get("severity", "none")),
            is_regression=bool(data.get("is_regression", False)),
            is_significant=bool(data.get("is_significant", False)),
            within_noise_band=bool(data.get("within_noise_band", False)),
            blocking=bool(data.get("blocking", False)),
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
    # The statistical policy the decision used (see the module docstring).
    alpha: float | None = None
    multiplicity: str | None = None
    family_size: int = 0
    suite: list[SuiteGateResult] = field(default_factory=list)

    @classmethod
    def not_requested(cls) -> GateResult:
        """The record for a run that had no ``--baseline-check``."""
        return cls(status=GateStatus.NOT_REQUESTED, exit_code=0)

    @property
    def requested(self) -> bool:
        return self.status is not GateStatus.NOT_REQUESTED

    def tasks_with(self, outcome: TaskGateOutcome) -> list[TaskGateResult]:
        return [t for t in self.tasks if t.outcome is outcome]

    @property
    def underpowered_tasks(self) -> list[TaskGateResult]:
        """Checked tasks with an observed drop the evidence could not confirm."""
        return [
            t for t in self.tasks
            if t.outcome is TaskGateOutcome.CHECKED and t.underpowered_regressions
        ]

    def policy_text(self) -> str:
        """The significance policy in one phrase."""
        if self.alpha is None:
            return "significance policy not recorded"
        if self.multiplicity == "holm":
            return (
                f"alpha={self.alpha:g}, Holm-adjusted across {self.family_size} "
                "checked task(s)"
            )
        return f"alpha={self.alpha:g} per task, no multiplicity correction"

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
        underpowered = sum(len(t.underpowered_regressions) for t in self.underpowered_tasks)
        if underpowered:
            parts.append(f"{underpowered} observed drop(s) not significant")
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
            "alpha": self.alpha,
            "multiplicity": self.multiplicity,
            "family_size": self.family_size,
            "checked": self.checked,
            "skipped_no_baseline": self.skipped_no_baseline,
            "skipped_no_gradable": self.skipped_no_gradable,
            "skipped_no_comparable_metrics": self.skipped_no_comparable_metrics,
            "skipped_task_content_changed": self.skipped_task_content_changed,
            "blocking_regressions": self.blocking_regressions,
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
            "suite": [s.to_dict() for s in self.suite],
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
            alpha=data.get("alpha"),
            multiplicity=data.get("multiplicity"),
            family_size=int(data.get("family_size", 0)),
            checked=int(data.get("checked", 0)),
            skipped_no_baseline=int(data.get("skipped_no_baseline", 0)),
            skipped_no_gradable=int(data.get("skipped_no_gradable", 0)),
            skipped_no_comparable_metrics=int(data.get("skipped_no_comparable_metrics", 0)),
            skipped_task_content_changed=int(data.get("skipped_task_content_changed", 0)),
            blocking_regressions=int(data.get("blocking_regressions", 0)),
            reasons=list(data.get("reasons", [])),
            warnings=list(data.get("warnings", [])),
            suite=[SuiteGateResult.from_dict(s) for s in data.get("suite", [])],
            tasks=[TaskGateResult.from_dict(t) for t in data.get("tasks", [])],
        )


def _apply_multiplicity(
    tasks: Sequence[TaskGateResult],
    detector: RegressionDetector,
    *,
    alpha: float,
    multiplicity: str,
) -> dict[str, float]:
    """Hold every checked task's findings to the run-level policy.

    Under ``holm`` the regression p-values of each metric are adjusted
    across the checked tasks that compared that metric (a task without a
    finding for it counts as a test with p = 1). Returns, per metric, the
    raw level one test must reach to be significant: ``alpha / m`` for a
    family of ``m`` tasks, or ``alpha`` without a correction.
    """
    checked = [t for t in tasks if t.outcome is TaskGateOutcome.CHECKED]
    metrics = sorted({m for t in checked for m in t.compared_metrics})
    levels: dict[str, float] = {}
    for metric in metrics:
        family = [t for t in checked if metric in t.compared_metrics]
        if multiplicity == "holm" and family:
            levels[metric] = alpha / len(family)
            findings: list[MetricRegression | None] = []
            for task in family:
                found = [r for r in task.regressions if r.metric_name == metric]
                findings.append(found[0] if found else None)
            raw = [
                1.0 if f is None or f.p_value is None else f.p_value for f in findings
            ]
            for finding, adjusted in zip(findings, holm_adjusted(raw), strict=True):
                if finding is not None and finding.p_value is not None:
                    finding.p_value_adjusted = adjusted
        else:
            levels[metric] = alpha
    for task in checked:
        for finding in task.regressions:
            detector.annotate(finding, alpha, levels.get(finding.metric_name, alpha))
        for finding in task.improvements:
            detector.annotate(finding, alpha)
    return levels


def _suite_results(
    per_task_means: Mapping[str, list[tuple[float, float]]],
    *,
    alpha: float,
    min_delta_percent: float,
    threshold: RegressionSeverity,
    noise_band: float,
    any_infra_mismatch: bool,
    seed: int,
) -> list[SuiteGateResult]:
    """The suite-level criterion per metric.

    ``per_task_means`` maps a metric to ``(baseline_mean, current_mean)``
    pairs, one per checked task that carries it.
    """
    results: list[SuiteGateResult] = []
    for metric, pairs in sorted(per_task_means.items()):
        if len(pairs) < 2:
            continue
        diffs = [current - baseline for baseline, current in pairs]
        effect = paired_task_effect(
            diffs, confidence=_SUITE_CONFIDENCE, n_bootstrap=_SUITE_BOOTSTRAP, seed=seed
        )
        baseline_mean = sum(b for b, _ in pairs) / len(pairs)
        current_mean = sum(c for _, c in pairs) / len(pairs)
        delta = effect.delta if effect.delta is not None else current_mean - baseline_mean
        delta_percent = relative_change(delta, baseline_mean)
        # The sign-flip distribution is symmetric, so the one-sided p-value
        # in the regression direction is half the two-sided one.
        p_one: float | None = None
        if effect.p_value is not None:
            p_one = effect.p_value / 2 if delta < 0 else 1.0 - effect.p_value / 2
            p_one = min(1.0, max(0.0, p_one))
        is_regression = delta < 0 and abs(delta_percent) >= min_delta_percent
        severity = severity_for(delta_percent) if is_regression else RegressionSeverity.NONE
        is_significant = is_regression and p_one is not None and p_one <= alpha
        within_noise = is_regression and any_infra_mismatch and abs(delta) < noise_band
        blocking = is_significant and not within_noise and severity_at_least(severity, threshold)
        results.append(SuiteGateResult(
            metric_name=metric,
            tasks=len(pairs),
            baseline_mean=baseline_mean,
            current_mean=current_mean,
            delta=delta,
            delta_percent=delta_percent,
            ci_lower=effect.ci_lower,
            ci_upper=effect.ci_upper,
            p_value=p_one,
            p_value_exact=effect.p_value_exact,
            confidence=effect.confidence,
            n_bootstrap=effect.n_bootstrap,
            seed=effect.seed,
            severity=severity,
            is_regression=is_regression,
            is_significant=is_significant,
            within_noise_band=within_noise,
            blocking=blocking,
        ))
    return results


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
    multiplicity: str = "holm",
    alpha: float = DEFAULT_SIGNIFICANCE_LEVEL,
    seed: int = 0,
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
        multiplicity: ``"holm"`` (default) adjusts each metric's per-task
            p-values across the checked tasks so the run's false-alarm rate
            from chance is at most ``alpha``; ``"none"`` holds every task to
            ``alpha`` on its own.
        alpha: Significance level for both criteria.
        seed: Seed of the suite-level bootstrap and sign-flip test.

    Returns:
        A :class:`GateResult` with one :class:`TaskGateResult` per task and
        one :class:`SuiteGateResult` per metric with two or more tasks.
    """
    if multiplicity not in MULTIPLICITY_CHOICES:
        raise ValueError(
            f"multiplicity must be one of {', '.join(MULTIPLICITY_CHOICES)}; got {multiplicity!r}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    # Power notes are computed once, after the run-level correction.
    detector = RegressionDetector(
        significance_level=alpha, noise_band_absolute=noise_band, power_notes=False
    )
    trials_by_task: dict[str, list[Trial]] = {}
    for trial in batch.trials:
        trials_by_task.setdefault(trial.task_id, []).append(trial)
    ordered = list(task_ids) if task_ids is not None else sorted(trials_by_task)
    if task_hashes is None:
        task_hashes = (
            batch.provenance.measurement.task_hashes if batch.provenance is not None else {}
        )
    unhashed_baselines: list[str] = []
    per_task_means: dict[str, list[tuple[float, float]]] = {}
    checked_inputs: dict[str, tuple[Any, list[dict[str, float]]]] = {}

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
        for metric in current_metrics:
            metric_baseline = baseline.get_metric(metric)
            if metric_baseline is None:
                continue
            values = [r[metric] for r in current_results if metric in r]
            per_task_means.setdefault(metric, []).append(
                (float(metric_baseline.baseline_value), sum(values) / len(values))
            )
        checked_inputs[task_id] = (baseline, current_results)
        tasks.append(TaskGateResult(
            task_id=task_id,
            outcome=TaskGateOutcome.CHECKED,
            compared_trials=len(current_results),
            excluded_trials=excluded,
            available_metrics=current_metrics,
            compared_metrics=[m for m in current_metrics if m in baseline.metrics],
            infra_config_mismatch=report.infra_config_mismatch,
            infra_config_diff=dict(report.infra_config_diff),
            regressions=list(report.regressions),
            improvements=list(report.improvements),
        ))

    # Run-level policy: adjust for multiplicity, then decide each task from
    # its significant findings and record what its sample sizes can show.
    levels = _apply_multiplicity(tasks, detector, alpha=alpha, multiplicity=multiplicity)
    checked = [t for t in tasks if t.outcome is TaskGateOutcome.CHECKED]
    family_size = max(
        (sum(1 for t in checked if metric in t.compared_metrics) for metric in levels),
        default=0,
    )
    # The strictest level any test in the run is held to, for the messages.
    per_test_level = min(levels.values(), default=alpha)
    for task in checked:
        report = task.regression_report()
        report.recompute()
        task.has_regression = report.has_regression
        task.overall_severity = report.overall_severity
        task.blocking = report.should_block_ci(threshold)
        baseline, current_results = checked_inputs[task.task_id]
        detectable = False
        needed: list[int] = []
        for metric in task.compared_metrics:
            metric_baseline = baseline.get_metric(metric)
            if metric_baseline is None:
                continue
            values = [r[metric] for r in current_results if metric in r]
            ok, trials = detector.detectability(
                metric_baseline, values, levels.get(metric, alpha)
            )
            detectable = detectable or ok
            if trials is not None:
                needed.append(trials)
        task.detectable = detectable
        task.trials_needed = None if detectable or not needed else min(needed)

    suite = _suite_results(
        per_task_means,
        alpha=alpha,
        min_delta_percent=detector.min_delta_percent,
        threshold=threshold,
        noise_band=noise_band,
        any_infra_mismatch=any(t.infra_config_mismatch for t in checked),
        seed=seed,
    )

    no_baseline = [t for t in tasks if t.outcome is TaskGateOutcome.NO_BASELINE]
    no_gradable = [t for t in tasks if t.outcome is TaskGateOutcome.NO_GRADABLE_TRIALS]
    no_comparable = [t for t in tasks if t.outcome is TaskGateOutcome.NO_COMPARABLE_METRICS]
    content_changed = [
        t for t in tasks if t.outcome is TaskGateOutcome.TASK_CONTENT_CHANGED
    ]
    blocking = [t for t in checked if t.blocking]
    suite_blocking = [s for s in suite if s.blocking]
    undetectable = [t for t in checked if not t.detectable]
    # The suite criterion can reject only when enough tasks could all move
    # the same way: the sign-flip p-value is at best 2^-T.
    suite_can_reject = any(2.0 ** -s.tasks <= alpha for s in suite)
    if unhashed_baselines:
        warnings.append(
            f"{len(unhashed_baselines)} baseline(s) carry no task_hash, so a change to "
            "their task content cannot be detected: " + ", ".join(unhashed_baselines)
            + "; re-store them from a results file that records provenance"
        )
    if undetectable and len(undetectable) < len(checked):
        needed = [t.trials_needed for t in undetectable if t.trials_needed is not None]
        advice = (
            f"; run at least {max(needed)} trials per task (and store baselines from "
            "at least as many)" if needed else ""
        )
        warnings.append(
            f"{len(undetectable)} checked task(s) have too few trials to block on their "
            f"own at {per_test_level:.4g} per test: "
            + ", ".join(t.task_id for t in undetectable) + advice
        )

    reasons: list[str] = []
    if (
        not checked
        or no_gradable
        or no_comparable
        or content_changed
        or (undetectable and len(undetectable) == len(checked) and not suite_can_reject)
    ):
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
        if checked and undetectable and len(undetectable) == len(checked) and not suite_can_reject:
            needed = [t.trials_needed for t in undetectable if t.trials_needed is not None]
            advice = (
                f"; run at least {max(needed)} trials per task and store baselines from "
                "at least as many" if needed else ""
            )
            reasons.append(
                "no checked task has enough trials to detect even a total failure "
                f"({per_test_level:.4g} per test, {len(checked)} task(s) checked), so the "
                "check could not have blocked" + advice
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
        if suite_blocking:
            status = GateStatus.BLOCKED
            reasons.append(
                f"{len(suite_blocking)} suite-level regression(s) at threshold "
                f"'{threshold.value}': " + "; ".join(s.describe() for s in suite_blocking)
            )
        if status is GateStatus.PASSED:
            reasons.append(
                f"{len(checked)} task(s) compared; no significant regression at or "
                f"above '{threshold.value}'"
            )
            underpowered = [t for t in checked if t.underpowered_regressions]
            if underpowered:
                reasons.append(
                    f"{len(underpowered)} task(s) show a drop the evidence could not "
                    "confirm: " + ", ".join(t.task_id for t in underpowered)
                    + "; see their notes for the trials needed"
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
        blocking_regressions=len(blocking) + len(suite_blocking),
        reasons=reasons,
        warnings=warnings,
        tasks=tasks,
        alpha=alpha,
        multiplicity=multiplicity,
        family_size=family_size,
        suite=suite,
    )
