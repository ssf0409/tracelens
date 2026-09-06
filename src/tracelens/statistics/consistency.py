"""pass^k (consistency) metric implementation.

pass^k is the fraction of windows of k consecutive runs in which every run
passed. Higher values indicate more reliable/consistent performance.

This is a reliability metric - it answers "is the agent consistent?"
Higher k values require longer streaks, so pass^k decreases with k.

Two properties matter for reading it correctly:

- It is a *consecutive-window* statistic over the run sequence, not
  ``pass_rate ** k`` and not an estimate of the probability that k
  independent attempts all succeed. Order matters: ``[T, T, F, F]`` and
  ``[T, F, T, F]`` have the same pass rate but different pass^2.
- The sequence must therefore be in ``run_index`` order, never in the order
  trials happened to finish. ``TrialBatch.get_pass_sequences_by_task``
  provides run-ordered sequences with ``None`` marking a missing run; a
  window that would span a gap is not counted.

See ``docs/statistical-contract.md`` for the full contract.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from tracelens.statistics.availability import MetricValue, unavailable_reason

PassSequence = Sequence[bool | None]
"""Pass/fail outcomes in run order; ``None`` marks a missing or excluded run."""


def _window_counts(results: PassSequence, k: int) -> tuple[int, int]:
    """Return ``(consistent_windows, complete_windows)`` for length-k windows.

    A window is *complete* when none of its entries is ``None``. Only
    complete windows are counted, in the numerator and denominator alike,
    so runs on either side of a gap are never treated as consecutive.
    """
    n = len(results)
    if n < k:
        return 0, 0
    consistent = 0
    complete = 0
    for i in range(n - k + 1):
        window = results[i : i + k]
        if any(r is None for r in window):
            continue
        complete += 1
        if all(window):
            consistent += 1
    return consistent, complete


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k!r}")


def pass_to_k(results: PassSequence, k: int) -> float:
    """Calculate pass^k (consistency) for one run sequence.

    ``results`` must be in run order (``run_index`` ascending). ``None``
    marks a run that is missing or excluded; windows that would include it
    are not counted.

    Args:
        results: Pass/fail booleans in run order, ``None`` for a missing run
        k: Number of consecutive passes required (at least 1)

    Returns:
        Fraction of complete k-length windows in which every run passed
        (0.0 to 1.0). Returns 0.0 when no complete window exists (fewer
        than k runs, or a gap in every window). The statistical contract
        treats that case as "unavailable"; the explicit N/A representation
        is tracked in issue #46.

    Raises:
        ValueError: If ``k`` is less than 1.

    Example:
        >>> pass_to_k([True, True, True, True, True], 3)
        1.0  # all 3 windows pass

        >>> pass_to_k([True, True, False, True, True], 3)
        0.0  # windows TTF, TFT, FTT: none is all-pass

        >>> pass_to_k([True, True, False, True, True], 2)
        0.5  # windows TT, TF, FT, TT: 2 of 4 pass

        >>> pass_to_k([True, True, None, True, True], 2)
        1.0  # windows spanning the gap are not counted; TT and TT remain
    """
    _check_k(k)
    consistent, complete = _window_counts(results, k)
    return consistent / complete if complete else 0.0


def pass_to_k_estimator(
    results_per_task: Mapping[str, PassSequence],
    k: int,
) -> float:
    """Compute suite pass^k: the mean of per-task pass^k over eligible tasks.

    A task is eligible when its run sequence contains at least one complete
    window of length k (at least k runs, with no gap inside the window).
    Ineligible tasks are dropped from the mean; explicit eligible/total
    counts are tracked in issue #46.

    Args:
        results_per_task: Dict mapping task_id to its pass/fail sequence in
            run order (``None`` for a missing run)
        k: Number of consecutive passes required (at least 1)

    Returns:
        Mean pass^k over eligible tasks, or 0.0 when no task is eligible.

    Raises:
        ValueError: If ``k`` is less than 1.

    Example:
        >>> results = {
        ...     "task1": [True, True, True, True, True],
        ...     "task2": [True, True, False, True, True],
        ... }
        >>> pass_to_k_estimator(results, k=3)
        0.5  # task1: 1.0, task2: 0.0 (no all-pass window of 3)
    """
    _check_k(k)
    scores = []
    for results in results_per_task.values():
        consistent, complete = _window_counts(results, k)
        if complete:
            scores.append(consistent / complete)
    return float(np.mean(scores)) if scores else 0.0


def pass_to_k_metric(
    results_per_task: Mapping[str, PassSequence],
    k: int,
) -> MetricValue:
    """Suite pass^k with explicit availability.

    Returns a :class:`MetricValue` whose ``value`` is ``None`` when no task
    has a complete window of ``k`` consecutive gradable runs, with the
    eligible/total task counts and the largest number of gradable runs any
    task recorded.

    Raises:
        ValueError: If ``k`` is less than 1.
    """
    _check_k(k)
    scores = []
    max_runs = 0
    for results in results_per_task.values():
        max_runs = max(max_runs, sum(1 for r in results if r is not None))
        consistent, complete = _window_counts(results, k)
        if complete:
            scores.append(consistent / complete)
    total = len(results_per_task)
    name = f"pass^{k}"
    if scores:
        return MetricValue(
            name=name,
            value=float(np.mean(scores)),
            eligible_tasks=len(scores),
            total_tasks=total,
            required_runs=k,
            max_runs=max_runs,
        )
    return MetricValue(
        name=name,
        value=None,
        eligible_tasks=0,
        total_tasks=total,
        required_runs=k,
        max_runs=max_runs,
        reason=unavailable_reason("consecutive gradable runs", k, total),
    )


class ConsistencyAnalyzer:
    """Analyzer for pass^k consistency metrics.

    Computes pass^k for multiple k values and provides reliability scoring.
    All methods take run-ordered sequences (see :data:`PassSequence`), as
    produced by ``TrialBatch.get_pass_sequences_by_task``.

    Example:
        analyzer = ConsistencyAnalyzer(k_values=[2, 3, 5])
        results = analyzer.analyze(pass_sequences_by_task)
        print(results)  # {"pass^2": 0.8, "pass^3": 0.6, "pass^5": 0.3}
    """

    def __init__(self, k_values: list[int] | None = None):
        """Initialize analyzer with k values to compute.

        Args:
            k_values: List of k values for pass^k. Default: [2, 3, 5]
        """
        self.k_values = k_values or [2, 3, 5]

    def analyze(
        self,
        results_per_task: Mapping[str, PassSequence],
    ) -> dict[str, float]:
        """Compute pass^k for multiple k values.

        Unavailable metrics (no task with a complete window of ``k`` runs)
        come back as ``0.0`` in this float API; :meth:`analyze_detailed`
        distinguishes them.

        Args:
            results_per_task: Dict mapping task_id to its run-ordered
                pass/fail sequence

        Returns:
            Dict mapping "pass^k" to computed value
        """
        return {
            f"pass^{k}": pass_to_k_estimator(results_per_task, k)
            for k in self.k_values
        }

    def analyze_detailed(
        self,
        results_per_task: Mapping[str, PassSequence],
    ) -> dict[str, MetricValue]:
        """Compute pass^k for multiple k values with availability evidence.

        Returns:
            Dict mapping "pass^k" to a :class:`MetricValue`; ``value`` is
            ``None`` where no task supports that ``k``.
        """
        return {
            f"pass^{k}": pass_to_k_metric(results_per_task, k)
            for k in self.k_values
        }

    def compute_reliability_score(
        self,
        results_per_task: Mapping[str, PassSequence],
    ) -> float:
        """Compute overall reliability score.

        Combines pass^k metrics weighted by k to give higher weight
        to longer consistent runs. A higher score indicates more
        reliable/consistent performance.

        Args:
            results_per_task: Dict mapping task_id to its run-ordered
                pass/fail sequence

        Returns:
            Weighted reliability score (0.0 to 1.0)
        """
        if not results_per_task:
            return 0.0

        weights = {k: k for k in self.k_values}
        total_weight = sum(weights.values())

        if total_weight == 0:
            return 0.0

        score = 0.0
        for k in self.k_values:
            pass_k = pass_to_k_estimator(results_per_task, k)
            score += weights[k] * pass_k

        return score / total_weight

    def compute_stability_metrics(
        self,
        results_per_task: Mapping[str, PassSequence],
    ) -> dict[str, float]:
        """Compute additional stability metrics.

        Returns:
            Dict with:
            - "pass^k" values for each k
            - "reliability_score": weighted combination
            - "failure_rate": failed share of observed runs (gaps not counted)
            - "avg_longest_streak": mean longest passing streak per task (a gap breaks it)
        """
        metrics = self.analyze(results_per_task)
        metrics["reliability_score"] = self.compute_reliability_score(results_per_task)

        # Failure rate over observed runs only.
        observed = [
            r for results in results_per_task.values() for r in results if r is not None
        ]
        if observed:
            metrics["failure_rate"] = 1.0 - (sum(observed) / len(observed))
        else:
            metrics["failure_rate"] = 1.0

        # Average longest streak; False and None both reset the streak.
        streaks = []
        for results in results_per_task.values():
            longest = 0
            current = 0
            for r in results:
                if r:
                    current += 1
                    longest = max(longest, current)
                else:
                    current = 0
            streaks.append(longest)

        metrics["avg_longest_streak"] = float(np.mean(streaks)) if streaks else 0.0

        return metrics
