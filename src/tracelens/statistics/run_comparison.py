"""Run-versus-run comparison: the paired task bootstrap (issue #28).

Implements the "Run-versus-run comparison" section of
``docs/statistical-contract.md``. Given two ``TrialBatch`` artifacts of the
same eval set, it aligns tasks by content (through their provenance),
computes one per-task statistic per run, takes the paired difference per
task, and reports the mean difference with a percentile bootstrap interval
over tasks, a paired sign-flip p-value, and a verdict against a practical
threshold. The task is the sampling unit; repeated trials of a task are
averaged into its statistic and never counted as independent samples.
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
    Verdict.INSUFFICIENT_EVIDENCE: "insufficient evidence (fewer than 2 paired tasks)",
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
    if t <= _EXACT_SIGN_FLIP_MAX_TASKS and 2**t <= n_bootstrap:
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
    if not 0.0 < confidence < 1.0:
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


def decide(effect: PairedEffect, threshold: float) -> Verdict:
    """Apply the contract's verdict table."""
    if effect.delta is None or effect.ci_lower is None or effect.ci_upper is None:
        return Verdict.INSUFFICIENT_EVIDENCE
    delta, lo, hi = effect.delta, effect.ci_lower, effect.ci_upper
    if excludes_zero(lo, hi):
        if delta <= -threshold:
            return Verdict.REGRESSION
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
    threshold: float
    significant: bool | None = None
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
            readings = [
                "significant" if self.significant else "not significant",
                (
                    f"|delta| >= threshold {self.threshold:g}"
                    if self.meaningful
                    else f"|delta| < threshold {self.threshold:g}"
                ),
            ]
            lines.append("  readings: " + ", ".join(readings))
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


def _what_changed(compat: CompatibilityReport) -> str:
    if compat.status is Compatibility.UNKNOWN:
        return "unknown (no provenance on one side)"
    if not compat.candidate_changed:
        return "nothing declared (same adapter and DecisionSpec fingerprint)"
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
    significant = (
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
        threshold=threshold,
        significant=significant,
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
    "compare_runs",
    "decide",
    "excludes_zero",
    "paired_task_effect",
    "short_hash",
]
