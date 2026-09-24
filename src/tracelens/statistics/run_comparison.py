"""Run-versus-run comparison: the paired task bootstrap (issue #28).

Implements the "Run-versus-run comparison" section of
``docs/statistical-contract.md``. Given two ``TrialBatch`` artifacts of the
same eval set, it aligns tasks by content (through their provenance),
computes one per-task statistic per run, takes the paired difference per
task, and reports the mean difference with a percentile bootstrap interval
over tasks, a paired sign-flip p-value, and a verdict against a practical
threshold. The task is the sampling unit; repeated trials of a task are
averaged into its statistic and never counted as independent samples.

A difference is significant only when the interval and the p-value agree
(issue #112). On a handful of tasks the percentile interval is too narrow,
while the sign-flip test is exact under the null of no change. Below the
number of tasks at which that test can reach the level at all there is no
verdict, and the output says how many tasks it would take.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, Field

from tracelens.core.outcome import Outcome
from tracelens.core.provenance import (
    Compatibility,
    CompatibilityReport,
    check_compatibility,
    short_hash,
)
from tracelens.core.trial import Trial, TrialBatch, TrialStatus

METHOD = "paired task bootstrap"
UNIT = "task"
DEFAULT_THRESHOLD = 0.03
BUILTIN_METRICS = ("pass_rate", "mean_score")
UNMATCHED_POLICIES = ("error", "exclude")
_EXACT_SIGN_FLIP_MAX_TASKS = 12
# An interval bound this close to zero is treated as touching zero, so
# floating-point residue from averaging never decides significance.
_ZERO_TOLERANCE = 1e-9
# The level a p-value is held to is rounded to this many digits, so that
# 1 - 0.95 is 0.05 and not 0.05000000000000004.
_ALPHA_DIGITS = 12


class Direction(StrEnum):
    """Which way is better for the selected metric."""

    HIGHER = "higher"
    LOWER = "lower"


class Verdict(StrEnum):
    """The contract's six outcomes; see the verdict table in the contract."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    BELOW_THRESHOLD = "significant_below_threshold"
    EQUIVALENT = "equivalent_within_threshold"
    INCONCLUSIVE = "inconclusive"


VERDICT_EXIT_CODES: dict[Verdict, int] = {
    Verdict.INSUFFICIENT_EVIDENCE: 2,
    Verdict.REGRESSION: 1,
    Verdict.IMPROVEMENT: 0,
    Verdict.BELOW_THRESHOLD: 0,
    Verdict.EQUIVALENT: 0,
    Verdict.INCONCLUSIVE: 2,
}

VERDICT_TEXT: dict[Verdict, str] = {
    Verdict.INSUFFICIENT_EVIDENCE: "insufficient evidence",
    Verdict.REGRESSION: "REGRESSION",
    Verdict.IMPROVEMENT: "IMPROVEMENT",
    Verdict.BELOW_THRESHOLD: "significant, but below the practical threshold",
    Verdict.EQUIVALENT: "equivalent within the practical threshold",
    Verdict.INCONCLUSIVE: "inconclusive: more runs or tasks needed",
}


class ComparisonError(ValueError):
    """The comparison cannot be evaluated. The message is user-facing."""


# --- Estimand -----------------------------------------------------------------


def _outcome_for(trial: Trial, grader_id: str) -> Outcome | None:
    for outcome in trial.outcomes:
        if outcome.grader_id == grader_id:
            return outcome
    return None


@dataclass(frozen=True)
class MetricSelector:
    """One metric, one direction, optionally one grader (the estimand)."""

    name: str
    direction: Direction
    grader_id: str | None = None

    @classmethod
    def parse(
        cls,
        metric: str = "pass_rate",
        direction: str | None = None,
        grader: str | None = None,
    ) -> MetricSelector:
        """Validate ``metric`` / ``--direction`` / ``--grader`` into a selector.

        Raises:
            ComparisonError: unknown metric, a direction on a built-in metric,
                an invalid direction, or a ``--grader`` that contradicts the
                grader named in ``<grader_id>.<metric_name>``.
        """
        metric = metric.strip()
        if metric in BUILTIN_METRICS:
            if direction not in (None, Direction.HIGHER.value):
                raise ComparisonError(
                    f"{metric} is always higher-is-better; do not pass --direction"
                )
            return cls(metric, Direction.HIGHER, grader)
        grader_part, _, metric_part = metric.partition(".")
        if not grader_part.strip() or not metric_part.strip():
            raise ComparisonError(
                f"unknown metric {metric!r}: use pass_rate, mean_score, or "
                "<grader_id>.<metric_name>"
            )
        if grader is not None and grader != grader_part:
            raise ComparisonError(
                f"--grader {grader!r} conflicts with the grader in metric {metric!r}"
            )
        try:
            resolved = Direction(direction or Direction.HIGHER.value)
        except ValueError as exc:
            raise ComparisonError(
                f"--direction must be 'higher' or 'lower', got {direction!r}"
            ) from exc
        return cls(metric, resolved, grader_part)

    @property
    def sign(self) -> float:
        """Multiplier that makes a positive effect an improvement."""
        return 1.0 if self.direction is Direction.HIGHER else -1.0

    def describe(self) -> str:
        text = f"{self.name} ({self.direction.value} is better)"
        if self.grader_id and self.name in BUILTIN_METRICS:
            text += f", grader {self.grader_id}"
        return text

    def value(self, trial: Trial) -> float | None:
        """The trial's value for this metric, or ``None`` when it has none.

        Only gradable trials have values. For ``pass_rate`` a gradable trial
        without any outcome (a timeout, or an agent failure before grading)
        is a failure, also under ``--grader``; a trial graded by other
        graders but not the selected one has no value. ``mean_score`` and
        outcome metrics have no value when the outcome or metric is missing
        or not finite.
        """
        if not trial.is_gradable:
            return None
        if self.name == "pass_rate":
            if self.grader_id is None:
                return 1.0 if trial.passed else 0.0
            outcome = _outcome_for(trial, self.grader_id)
            if outcome is None:
                return 0.0 if not trial.outcomes else None
            return 1.0 if outcome.passed else 0.0
        if self.name == "mean_score":
            if self.grader_id is None:
                return _finite(trial.aggregate_score)
            outcome = _outcome_for(trial, self.grader_id)
            return None if outcome is None else _finite(float(outcome.score))
        grader_id, _, metric_name = self.name.partition(".")
        outcome = _outcome_for(trial, grader_id)
        if outcome is None or metric_name not in outcome.metrics:
            return None
        return _finite(float(outcome.metrics[metric_name]))


def _finite(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(value)


# --- Per-run extraction -------------------------------------------------------


class SideSummary(BaseModel):
    """Trial accounting for one side of the comparison."""

    label: str
    run_id: str | None = None
    trials_total: int
    trials_gradable: int
    excluded: dict[str, int] = Field(default_factory=dict)
    tasks_with_values: int

    def describe(self) -> str:
        excluded = ", ".join(f"{k} {v}" for k, v in self.excluded.items() if v)
        text = (
            f"{self.trials_total} trials, {self.trials_gradable} gradable, "
            f"{self.tasks_with_values} task(s) with values"
        )
        if excluded:
            text += f" (excluded: {excluded})"
        return text


def _extract(
    batch: TrialBatch, selector: MetricSelector, label: str
) -> tuple[dict[str, list[float]], SideSummary]:
    values: dict[str, list[float]] = {}
    excluded = {"infra_error": 0, "grader_error": 0, "not_run": 0, "no_value": 0}
    gradable = 0
    for trial in sorted(batch.trials, key=lambda t: (t.task_id, t.run_index)):
        if trial.status is TrialStatus.INFRA_ERROR:
            excluded["infra_error"] += 1
            continue
        if trial.has_grader_error:
            excluded["grader_error"] += 1
            continue
        if not trial.is_gradable:
            excluded["not_run"] += 1
            continue
        gradable += 1
        value = selector.value(trial)
        if value is None:
            excluded["no_value"] += 1
            continue
        values.setdefault(trial.task_id, []).append(value)
    summary = SideSummary(
        label=label,
        run_id=batch.provenance.run_id if batch.provenance is not None else batch.batch_id,
        trials_total=batch.total_count,
        trials_gradable=gradable,
        excluded=excluded,
        tasks_with_values=len(values),
    )
    return values, summary


# --- Paired statistics ---------------------------------------------------------


class PairedEffect(BaseModel):
    """Mean paired difference over tasks with its bootstrap interval and p-value."""

    tasks: int
    delta: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    p_value_exact: bool = False
    confidence: float
    n_bootstrap: int
    seed: int | None


def _bootstrap_means(diffs: np.ndarray, n_bootstrap: int, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = len(diffs)
    out = np.empty(n_bootstrap)
    chunk = max(1, min(n_bootstrap, 2_000_000 // t))
    start = 0
    while start < n_bootstrap:
        size = min(chunk, n_bootstrap - start)
        index = rng.integers(0, t, size=(size, t))
        out[start : start + size] = diffs[index].mean(axis=1)
        start += size
    return out


def _enumerates(tasks: int, n_bootstrap: int) -> bool:
    """Whether the sign-flip p-value enumerates every assignment, and so is exact."""
    return tasks <= _EXACT_SIGN_FLIP_MAX_TASKS and 2**tasks <= n_bootstrap


def alpha_for(confidence: float) -> float:
    """The level a p-value is held to: ``1 - confidence``, without float residue."""
    return round(1.0 - confidence, _ALPHA_DIGITS)


def min_attainable_p(tasks: int, n_bootstrap: int) -> float | None:
    """The smallest p-value the sign-flip test can return with ``tasks`` paired tasks.

    Only the observed sign assignment and its mirror image can be as extreme
    as the observed mean, and only when every difference is non-zero and all
    share one sign, so the exact two-sided p-value is never below ``2 / 2^T``.
    When the assignments are sampled rather than enumerated, the estimate
    ``(extreme + 1) / (B + 1)`` is also never below ``1 / (B + 1)``. ``None``
    without tasks.
    """
    if tasks < 1:
        return None
    floor = 2.0 ** (1 - tasks)
    if _enumerates(tasks, n_bootstrap):
        return floor
    return max(floor, 1.0 / (n_bootstrap + 1))


def min_tasks_for(confidence: float) -> int:
    """The fewest paired tasks with which the sign-flip test can reach the level.

    The smallest ``T`` with ``2 / 2^T <= 1 - confidence``: 6 at 0.95, 5 at
    0.90, 8 at 0.99. With fewer tasks no difference is significant however
    large and consistent it is, so there is no verdict.

    Raises:
        ValueError: If ``confidence`` is not strictly between 0 and 1.
    """
    alpha = alpha_for(confidence)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence!r}")
    tasks = 2
    while 2.0 ** (1 - tasks) > alpha:
        tasks += 1
    return tasks


def _sign_flip_p_value(
    diffs: np.ndarray, delta: float, n_bootstrap: int, seed: int | None
) -> tuple[float, bool]:
    """Two-sided paired sign-flip p-value.

    Exact (all ``2^T`` assignments) when ``T`` is small enough for that to
    cost no more than ``n_bootstrap`` draws; otherwise ``n_bootstrap`` random
    assignments, with the observed one counted.
    """
    t = len(diffs)
    observed = abs(delta) - 1e-12
    if _enumerates(t, n_bootstrap):
        patterns = np.arange(2**t)[:, None] >> np.arange(t)
        signs = (patterns & 1) * 2 - 1
        means = signs @ diffs / t
        return float(np.mean(np.abs(means) >= observed)), True
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = max(1, min(n_bootstrap, 2_000_000 // t))
    start = 0
    while start < n_bootstrap:
        size = min(chunk, n_bootstrap - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, t))
        means = signs @ diffs / t
        extreme += int(np.sum(np.abs(means) >= observed))
        start += size
    return (extreme + 1) / (n_bootstrap + 1), False


def paired_task_effect(
    diffs: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = 0,
) -> PairedEffect:
    """Mean of paired per-task differences with a task bootstrap and sign-flip test.

    ``diffs`` holds one direction-normalised difference per task (positive is
    an improvement). Fewer than two tasks yield no interval and no p-value.

    Raises:
        ValueError: If ``confidence`` is not strictly between 0 and 1, or
            ``n_bootstrap`` is less than 1.
    """
    if not 0.0 < alpha_for(confidence) < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence!r}")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be at least 1, got {n_bootstrap!r}")
    values = np.asarray(sorted(float(d) for d in diffs))
    t = len(values)
    effect = PairedEffect(tasks=t, confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
    if t == 0:
        return effect
    delta = float(values.mean())
    effect.delta = delta
    if t < 2:
        return effect
    means = _bootstrap_means(values, n_bootstrap, seed)
    alpha = (1.0 - confidence) / 2.0
    effect.ci_lower = float(np.percentile(means, alpha * 100))
    effect.ci_upper = float(np.percentile(means, (1.0 - alpha) * 100))
    effect.p_value, effect.p_value_exact = _sign_flip_p_value(values, delta, n_bootstrap, seed)
    return effect


def excludes_zero(lower: float, upper: float) -> bool:
    """Whether an interval excludes zero, ignoring floating-point residue."""
    return lower > _ZERO_TOLERANCE or upper < -_ZERO_TOLERANCE


def can_reach_level(effect: PairedEffect) -> bool:
    """Whether the sign-flip test could return ``p <= 1 - confidence`` at all.

    False with fewer than :func:`min_tasks_for` paired tasks, or with too few
    sampled sign assignments to resolve the level.
    """
    attainable = min_attainable_p(effect.tasks, effect.n_bootstrap)
    return attainable is not None and attainable <= alpha_for(effect.confidence)


def is_significant(effect: PairedEffect) -> bool | None:
    """The contract's significance reading: the interval and the p-value agree.

    Significant when the interval excludes 0 *and* the sign-flip p-value is
    at most ``1 - confidence`` (which requires enough tasks to reach it).
    ``None`` without an interval.
    """
    if effect.ci_lower is None or effect.ci_upper is None or effect.p_value is None:
        return None
    return (
        can_reach_level(effect)
        and excludes_zero(effect.ci_lower, effect.ci_upper)
        and effect.p_value <= alpha_for(effect.confidence)
    )


def decide(effect: PairedEffect, threshold: float) -> Verdict:
    """Apply the contract's verdict table.

    There is no verdict without enough tasks for the sign-flip test to reach
    the level. A difference is significant only when the interval and the
    p-value agree, and a significant difference below the threshold passes
    only when the interval also rules out a regression of the threshold or
    more, so every verdict that exits 0 has ``ci_lower > -threshold``.
    """
    significant = is_significant(effect)
    if significant is None or effect.delta is None or not can_reach_level(effect):
        return Verdict.INSUFFICIENT_EVIDENCE
    assert effect.ci_lower is not None and effect.ci_upper is not None  # as significant
    delta, lo, hi = effect.delta, effect.ci_lower, effect.ci_upper
    if significant:
        if delta <= -threshold:
            return Verdict.REGRESSION
        if lo <= -threshold:
            # Significant, and small on the estimate, but a regression of the
            # threshold or more is still inside the interval.
            return Verdict.INCONCLUSIVE
        if delta >= threshold:
            return Verdict.IMPROVEMENT
        return Verdict.BELOW_THRESHOLD
    if lo > -threshold and hi < threshold:
        return Verdict.EQUIVALENT
    return Verdict.INCONCLUSIVE


# --- The comparison ------------------------------------------------------------


class TaskAlignmentSummary(BaseModel):
    """How the two task sets were matched and what was left out."""

    aligned_by: str  # "content" (provenance hashes) or "id" (no provenance)
    compared: int
    excluded_changed: list[str] = Field(default_factory=list)
    excluded_only_baseline: list[str] = Field(default_factory=list)
    excluded_only_candidate: list[str] = Field(default_factory=list)
    excluded_no_value_baseline: list[str] = Field(default_factory=list)
    excluded_no_value_candidate: list[str] = Field(default_factory=list)

    def describe(self) -> str:
        parts = [f"{self.compared} task(s) compared, aligned by {self.aligned_by}"]
        for label, ids in (
            ("changed content", self.excluded_changed),
            ("only in baseline", self.excluded_only_baseline),
            ("only in candidate", self.excluded_only_candidate),
            ("no value in baseline", self.excluded_no_value_baseline),
            ("no value in candidate", self.excluded_no_value_candidate),
        ):
            if ids:
                parts.append(f"{len(ids)} excluded ({label}: {_list_ids(ids)})")
        return "; ".join(parts)


class TaskDelta(BaseModel):
    """One task's statistic on each side and its direction-normalised difference."""

    task_id: str
    baseline: float
    candidate: float
    delta: float
    n_baseline: int
    n_candidate: int


class RunComparison(BaseModel):
    """The full comparison record; the CLI summary and JSON share every field."""

    method: str = METHOD
    unit: str = UNIT
    metric: str
    direction: Direction
    grader: str | None = None
    baseline: SideSummary
    candidate: SideSummary
    alignment: TaskAlignmentSummary
    # ``delta`` and the interval are direction-normalised: positive is an
    # improvement. ``raw_delta`` is candidate minus baseline on the metric's
    # own scale (equal to ``delta`` for higher-is-better metrics).
    delta: float | None = None
    raw_delta: float | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence: float
    n_bootstrap: int
    seed: int | None
    p_value: float | None = None
    p_value_exact: bool = False
    # The smallest p-value the sign-flip test could return with these tasks
    # and draws, and the fewest tasks with which it can reach the level at
    # this confidence. Below that there is no verdict.
    min_attainable_p: float | None = None
    min_tasks: int | None = None
    threshold: float
    # ``significant`` is the contract's reading: the interval excludes 0 and
    # the p-value agrees. ``interval_excludes_zero`` is the interval alone.
    significant: bool | None = None
    interval_excludes_zero: bool | None = None
    meaningful: bool | None = None
    verdict: Verdict
    exit_code: int
    observe: bool = False
    per_task: list[TaskDelta] = Field(default_factory=list)
    compatibility: CompatibilityReport
    notes: list[str] = Field(default_factory=list)

    def summary_lines(self, top: int = 5) -> list[str]:
        """The terminal rendering: the same facts as the JSON, in reading order."""
        selector = MetricSelector(self.metric, self.direction, self.grader)
        lines = [
            f"Compared {self.candidate.label} vs {self.baseline.label} on "
            f"{selector.describe()}: {self.method} over {self.alignment.compared} task(s)",
            f"  {self.baseline.label}: {self.baseline.describe()}",
            f"  {self.candidate.label}: {self.candidate.describe()}",
            f"  tasks: {self.alignment.describe()}",
        ]
        if self.direction is Direction.LOWER and self.raw_delta is not None:
            lines.append(
                f"  raw delta (candidate - baseline) = {self.raw_delta:+.4f}; the values "
                "below are improvement-positive"
            )
        if self.delta is None:
            lines.append("  delta: n/a (no task has a value on both sides)")
        elif self.ci_lower is None or self.ci_upper is None:
            lines.append(f"  delta = {self.delta:+.4f}; no interval (fewer than 2 tasks)")
        else:
            p_text = (
                "n/a"
                if self.p_value is None
                else f"{self.p_value:.4f}" + (" (exact)" if self.p_value_exact else "")
            )
            lines.append(
                f"  delta = {self.delta:+.4f}  {self.confidence:.0%} CI "
                f"[{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]  p = {p_text}  "
                f"(B = {self.n_bootstrap}, seed = {self.seed})"
            )
            lines.append("  readings: " + ", ".join(self._readings(self.ci_lower, self.ci_upper)))
        if self.verdict is Verdict.INSUFFICIENT_EVIDENCE and self.min_attainable_p is not None:
            lines.append("  evidence: " + self._shortfall(self.min_attainable_p))
        lines.append(f"  Verdict: {VERDICT_TEXT[self.verdict]} (exit {self.exit_code})")
        lines.append("  What changed: " + _what_changed(self.compatibility))
        moved = [row for row in self.per_task if row.delta != 0.0]
        if self.per_task and not moved:
            lines.append("  What moved: nothing; every compared task has the same value")
        elif moved:
            movers = ", ".join(
                f"{row.task_id} {row.delta:+.3f} (n {row.n_baseline}/{row.n_candidate})"
                for row in moved[:top]
            )
            more = len(moved) - top
            lines.append(
                "  What moved (largest first): " + movers
                + (f", and {more} more" if more > 0 else "")
            )
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return lines

    def _readings(self, lower: float, upper: float) -> list[str]:
        """Significance, practical relevance, and the interval's extent against the threshold."""
        alpha = alpha_for(self.confidence)
        reachable = self.min_attainable_p is not None and self.min_attainable_p <= alpha
        p_agrees = reachable and self.p_value is not None and self.p_value <= alpha
        if self.significant:
            significance = "significant"
        elif self.interval_excludes_zero and not reachable:
            significance = "not significant (the interval excludes 0, but too few tasks for p)"
        elif self.interval_excludes_zero:
            significance = f"not significant (the interval excludes 0, but p > {alpha:g})"
        elif p_agrees:
            significance = f"not significant (p <= {alpha:g}, but the interval includes 0)"
        else:
            significance = "not significant"
        tau = f"{self.threshold:g}"
        relevance = (
            f"|delta| >= threshold {tau}" if self.meaningful else f"|delta| < threshold {tau}"
        )
        low, high = lower <= -self.threshold, upper >= self.threshold
        if low and high:
            extent = f"interval reaches both -{tau} and +{tau}"
        elif upper <= -self.threshold:
            extent = f"interval beyond -{tau}"
        elif lower >= self.threshold:
            extent = f"interval beyond +{tau}"
        elif low:
            extent = f"interval reaches -{tau}"
        elif high:
            extent = f"interval reaches +{tau}"
        else:
            extent = f"interval inside (-{tau}, +{tau})"
        return [significance, relevance, extent]

    def _shortfall(self, attainable: float) -> str:
        """Why there is no verdict: the p-value the test cannot get below."""
        alpha = alpha_for(self.confidence)
        need = f"a {self.confidence * 100:g}% verdict needs p <= {alpha:g}"
        tasks = self.alignment.compared
        if self.min_tasks is not None and tasks < self.min_tasks:
            return (
                f"with {tasks} paired task(s) the sign-flip test cannot give p below "
                f"{attainable:.4f}; {need}, which takes at least {self.min_tasks} tasks"
            )
        return (
            f"with B = {self.n_bootstrap} sampled sign flips the test cannot give p below "
            f"{attainable:.4f}; {need}, which takes B >= {math.ceil(1 / alpha) - 1}"
        )


def _what_changed(compat: CompatibilityReport) -> str:
    if compat.status is Compatibility.UNKNOWN:
        return "unknown (no provenance on one side)"
    if compat.candidate_changed is None:
        # The declared identity matches but a source hash is missing on one
        # side (an artifact written before it was recorded), so an edit that
        # kept the class path cannot be ruled out. Say so; do not call it
        # unchanged.
        return (
            "nothing declared (same adapter class path and DecisionSpec fingerprint; "
            "adapter source not recorded on one side, so a code edit would not show)"
        )
    if not compat.candidate_changed:
        return "nothing (same adapter source and DecisionSpec fingerprint)"
    parts = []
    if compat.adapter_changed:
        parts.append("adapter")
    if compat.candidate_diff:
        parts.append("DecisionSpec " + ", ".join(sorted(compat.candidate_diff)))
    elif not compat.adapter_changed:
        parts.append("DecisionSpec fingerprint")
    return ", ".join(parts) + " (attribution evidence, not proof of cause)"


def _list_ids(ids: Sequence[str], limit: int = 8) -> str:
    shown = ", ".join(ids[:limit])
    if len(ids) > limit:
        shown += f", and {len(ids) - limit} more"
    return shown


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def compare_runs(
    baseline: TrialBatch,
    candidate: TrialBatch,
    *,
    metric: str = "pass_rate",
    direction: str | None = None,
    grader: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = 0,
    unmatched_tasks: str = "error",
    require_provenance: bool = False,
    observe: bool = False,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
) -> RunComparison:
    """Compare two saved runs per the statistical contract.

    Args:
        baseline: The reference run (``tracelens run --save-trials``).
        candidate: The run under test.
        metric: ``pass_rate``, ``mean_score``, or ``<grader_id>.<metric_name>``.
        direction: ``higher`` or ``lower`` (custom metrics only; built-ins
            are higher-is-better).
        grader: Restrict ``pass_rate`` / ``mean_score`` to one grader's outcome.
        threshold: Practical threshold, an absolute delta on the metric scale.
        confidence: Interval confidence level.
        n_bootstrap: Bootstrap resamples (and sign-flip draws).
        seed: Seed for both procedures; same inputs and seed reproduce the
            result exactly.
        unmatched_tasks: ``error`` (default) refuses task-set differences;
            ``exclude`` compares the shared, unchanged tasks and lists the rest.
        require_provenance: Refuse artifacts without provenance instead of
            aligning their tasks by id.
        observe: Observational mode: every evaluated comparison exits 0.

    Raises:
        ComparisonError: Invalid selection, incompatible measurement setups
            (different graders; task-set differences under ``error``), or
            missing provenance under ``require_provenance``.
    """
    selector = MetricSelector.parse(metric, direction, grader)
    if threshold < 0:
        raise ComparisonError(f"threshold cannot be negative, got {threshold!r}")
    if unmatched_tasks not in UNMATCHED_POLICIES:
        raise ComparisonError(
            f"unmatched_tasks must be one of {', '.join(UNMATCHED_POLICIES)}, "
            f"got {unmatched_tasks!r}"
        )
    try:
        paired_task_effect([], confidence=confidence, n_bootstrap=n_bootstrap, seed=seed)
    except ValueError as exc:
        raise ComparisonError(str(exc)) from exc

    a_values, a_summary = _extract(baseline, selector, baseline_label)
    b_values, b_summary = _extract(candidate, selector, candidate_label)
    compat = check_compatibility(baseline.provenance, candidate.provenance)
    notes: list[str] = list(compat.notes)
    if selector.name not in BUILTIN_METRICS and threshold == DEFAULT_THRESHOLD:
        notes.append(
            f"threshold {DEFAULT_THRESHOLD} is the default for 0-1 metrics; set "
            f"--threshold on the scale of {selector.name}"
        )

    if compat.status is Compatibility.UNKNOWN:
        if require_provenance:
            raise ComparisonError(
                "; ".join(compat.reasons) + "; re-run with a TraceLens that records "
                "provenance, or drop --require-provenance to align tasks by id"
            )
        aligned_by = "id"
        a_ids = {t.task_id for t in baseline.trials}
        b_ids = {t.task_id for t in candidate.trials}
        same = sorted(a_ids & b_ids)
        changed: list[str] = []
        only_a = sorted(a_ids - b_ids)
        only_b = sorted(b_ids - a_ids)
        notes.append(
            "compatibility unknown (" + "; ".join(compat.reasons)
            + "); tasks aligned by id only, so a task edited between the runs "
            "would not be detected"
        )
    else:
        if compat.graders_changed:
            raise ComparisonError(
                "the runs were graded differently, so they do not measure the same "
                "thing: " + "; ".join(r for r in compat.reasons if r.startswith("graders"))
            )
        aligned_by = "content"
        assert compat.tasks is not None  # set whenever provenance exists on both sides
        same = list(compat.tasks.same)
        changed = list(compat.tasks.changed)
        only_a = list(compat.tasks.only_in_a)
        only_b = list(compat.tasks.only_in_b)

    if (changed or only_a or only_b) and unmatched_tasks == "error":
        problems = []
        if changed:
            problems.append(f"{len(changed)} changed content ({_list_ids(changed)})")
        if only_a:
            problems.append(f"{len(only_a)} only in baseline ({_list_ids(only_a)})")
        if only_b:
            problems.append(f"{len(only_b)} only in candidate ({_list_ids(only_b)})")
        raise ComparisonError(
            "task sets differ: " + "; ".join(problems)
            + f"; pass --unmatched-tasks exclude to compare the {len(same)} shared "
            "task(s) and list the rest"
        )

    no_value_a = [t for t in same if t not in a_values]
    no_value_b = [t for t in same if t not in b_values]
    compared = [t for t in same if t in a_values and t in b_values]
    rows = [
        TaskDelta(
            task_id=task_id,
            baseline=_mean(a_values[task_id]),
            candidate=_mean(b_values[task_id]),
            delta=selector.sign * (_mean(b_values[task_id]) - _mean(a_values[task_id])),
            n_baseline=len(a_values[task_id]),
            n_candidate=len(b_values[task_id]),
        )
        for task_id in compared
    ]
    effect = paired_task_effect(
        [row.delta for row in rows], confidence=confidence, n_bootstrap=n_bootstrap, seed=seed
    )
    verdict = decide(effect, threshold)
    exit_code = VERDICT_EXIT_CODES[verdict]
    if observe and effect.delta is not None:
        exit_code = 0
    rows.sort(key=lambda row: (-abs(row.delta), row.task_id))
    interval_excludes_zero = (
        None
        if effect.ci_lower is None or effect.ci_upper is None
        else excludes_zero(effect.ci_lower, effect.ci_upper)
    )
    return RunComparison(
        metric=selector.name,
        direction=selector.direction,
        grader=selector.grader_id if selector.name in BUILTIN_METRICS else None,
        baseline=a_summary,
        candidate=b_summary,
        alignment=TaskAlignmentSummary(
            aligned_by=aligned_by,
            compared=len(compared),
            excluded_changed=changed,
            excluded_only_baseline=only_a,
            excluded_only_candidate=only_b,
            excluded_no_value_baseline=no_value_a,
            excluded_no_value_candidate=no_value_b,
        ),
        delta=effect.delta,
        raw_delta=None if effect.delta is None else selector.sign * effect.delta,
        ci_lower=effect.ci_lower,
        ci_upper=effect.ci_upper,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        seed=seed,
        p_value=effect.p_value,
        p_value_exact=effect.p_value_exact,
        min_attainable_p=min_attainable_p(effect.tasks, n_bootstrap),
        min_tasks=min_tasks_for(confidence),
        threshold=threshold,
        significant=is_significant(effect),
        interval_excludes_zero=interval_excludes_zero,
        meaningful=None if effect.delta is None else abs(effect.delta) >= threshold,
        verdict=verdict,
        exit_code=exit_code,
        observe=observe,
        per_task=rows,
        compatibility=compat,
        notes=notes,
    )


__all__ = [
    "DEFAULT_THRESHOLD",
    "ComparisonError",
    "Direction",
    "MetricSelector",
    "PairedEffect",
    "RunComparison",
    "SideSummary",
    "TaskAlignmentSummary",
    "TaskDelta",
    "Verdict",
    "alpha_for",
    "can_reach_level",
    "compare_runs",
    "decide",
    "excludes_zero",
    "is_significant",
    "min_attainable_p",
    "min_tasks_for",
    "paired_task_effect",
    "short_hash",
]
