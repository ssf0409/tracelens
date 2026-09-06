"""pass@k metric implementation.

pass@k estimates the probability that at least one of k samples passes,
given n total samples with c correct.

This is a capability metric - it answers "can the agent do this at all?"
Higher k values give the agent more chances, so pass@k increases with k.

Suite-level numbers treat the *task* as the sampling unit: per-task pass@k
is computed once and the suite value is the mean over tasks. Bootstrap
confidence intervals resample tasks with replacement, preserving repeated
draws. See ``docs/statistical-contract.md`` for the full contract.

Reference: Chen et al., "Evaluating Large Language Models Trained on Code"
"""

import numpy as np

from tracelens.statistics.availability import MetricValue, unavailable_reason
from tracelens.statistics.inference import bootstrap_ci


def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k metric.

    Estimates the probability that at least one of k samples passes,
    given n total samples with c correct. Uses an unbiased estimator.

    Args:
        n: Total number of samples
        c: Number of correct/passing samples
        k: Number of samples to consider

    Returns:
        Probability of at least one pass in k samples (0.0 to 1.0)

    Example:
        >>> pass_at_k(10, 7, 5)
        0.9916...  # Very likely at least 1 of 5 passes

        >>> pass_at_k(10, 1, 5)
        0.5  # 50% chance at least 1 of 5 passes
    """
    if n - c < k:
        # More passes needed than failures available
        return 1.0

    if c == 0:
        # No passes, so pass@k is 0
        return 0.0

    # Unbiased estimator: 1 - C(n-c, k) / C(n, k)
    # Equivalent to: 1 - prod((n-c-i)/(n-i) for i in range(k))
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def _per_task_pass_at_k(
    results_per_task: dict[str, list[bool]],
    k: int,
) -> dict[str, float]:
    """Per-task pass@k scores keyed by task_id, in sorted task_id order.

    Only tasks with at least ``k`` gradable runs are eligible; the unbiased
    estimator is undefined below that, so such tasks are omitted rather than
    approximated (statistical contract: unavailable, never a fallback).

    Sorting by task_id gives every caller a canonical order, so seeded
    resampling does not depend on dict insertion order.
    """
    scores: dict[str, float] = {}
    for task_id in sorted(results_per_task):
        results = results_per_task[task_id]
        n = len(results)
        if n >= k:
            scores[task_id] = pass_at_k(n, sum(results), k)
    return scores


def pass_at_k_estimator(
    results_per_task: dict[str, list[bool]],
    k: int,
) -> float:
    """Compute suite pass@k: the mean of per-task pass@k over eligible tasks.

    A task is eligible when it has at least ``k`` gradable runs. This
    float-returning API keeps the legacy convention of ``0.0`` when no task
    is eligible; use :func:`pass_at_k_metric` to tell that apart from a
    measured zero.

    Args:
        results_per_task: Dict mapping task_id to list of pass/fail booleans
        k: Number of samples to consider

    Returns:
        Mean pass@k over eligible tasks, or 0.0 when no task is eligible

    Example:
        >>> results = {
        ...     "task1": [True, True, False, True, True],
        ...     "task2": [False, True, False, False, True],
        ... }
        >>> pass_at_k_estimator(results, k=3)
        0.9...  # High probability task1 passes, lower for task2
    """
    scores = list(_per_task_pass_at_k(results_per_task, k).values())
    return float(np.mean(scores)) if scores else 0.0


def pass_at_k_metric(
    results_per_task: dict[str, list[bool]],
    k: int,
) -> MetricValue:
    """Suite pass@k with explicit availability.

    Returns a :class:`MetricValue` whose ``value`` is ``None`` when no task
    has at least ``k`` gradable runs, with the eligible/total task counts
    and the largest run count recorded so a report can say why.

    Raises:
        ValueError: If ``k`` is less than 1.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k!r}")
    scores = _per_task_pass_at_k(results_per_task, k)
    total = len(results_per_task)
    max_runs = max((len(r) for r in results_per_task.values()), default=0)
    name = f"pass@{k}"
    if scores:
        return MetricValue(
            name=name,
            value=float(np.mean(list(scores.values()))),
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
        reason=unavailable_reason("gradable runs", k, total),
    )


class PassAtKAnalyzer:
    """Analyzer for pass@k capability metrics.

    Computes pass@k for multiple k values and provides confidence intervals.

    Example:
        analyzer = PassAtKAnalyzer(k_values=[1, 3, 5, 10])
        results = analyzer.analyze(pass_results_by_task)
        print(results)  # {"pass@1": 0.6, "pass@3": 0.85, "pass@5": 0.95, "pass@10": 0.99}
    """

    def __init__(self, k_values: list[int] | None = None):
        """Initialize analyzer with k values to compute.

        Args:
            k_values: List of k values for pass@k. Default: [1, 5, 10]
        """
        self.k_values = k_values or [1, 5, 10]

    def analyze(
        self,
        results_per_task: dict[str, list[bool]],
    ) -> dict[str, float]:
        """Compute pass@k for multiple k values.

        Unavailable metrics (no task with ``k`` gradable runs) come back as
        ``0.0`` in this float API; :meth:`analyze_detailed` distinguishes them.

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans

        Returns:
            Dict mapping "pass@k" to computed value
        """
        return {
            f"pass@{k}": pass_at_k_estimator(results_per_task, k)
            for k in self.k_values
        }

    def analyze_detailed(
        self,
        results_per_task: dict[str, list[bool]],
    ) -> dict[str, MetricValue]:
        """Compute pass@k for multiple k values with availability evidence.

        Returns:
            Dict mapping "pass@k" to a :class:`MetricValue`; ``value`` is
            ``None`` where no task supports that ``k``.
        """
        return {
            f"pass@{k}": pass_at_k_metric(results_per_task, k)
            for k in self.k_values
        }

    def compute_confidence_interval(
        self,
        results_per_task: dict[str, list[bool]],
        k: int,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> tuple[float, float]:
        """Compute a bootstrap confidence interval for suite pass@k.

        The sampling unit is the task. Per-task pass@k scores are computed
        once (in sorted task_id order), then resampled with replacement
        ``n_bootstrap`` times; a task drawn twice contributes twice. The
        interval is the percentile interval of the resampled suite means.

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans
            k: k value for pass@k
            confidence: Confidence level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap resamples
            seed: Seed for the resampling generator. The same inputs and
                seed always yield the same interval, and the order in which
                tasks appear in ``results_per_task`` does not affect it.

        Returns:
            Tuple of (lower_bound, upper_bound). With no tasks this is
            ``(0.0, 0.0)``, a legacy placeholder for "unavailable" (issue
            #46). With a single task the interval degenerates to that
            task's score; it carries no uncertainty information.

        Raises:
            ValueError: If ``confidence`` is not strictly between 0 and 1,
                or ``n_bootstrap`` is less than 1.
        """
        scores = list(_per_task_pass_at_k(results_per_task, k).values())
        _, lower, upper = bootstrap_ci(
            scores,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            statistic="mean",
            seed=seed,
        )
        return lower, upper

    def analyze_with_ci(
        self,
        results_per_task: dict[str, list[bool]],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
        seed: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """Compute pass@k with confidence intervals.

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans
            confidence: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap resamples
            seed: Seed for the resampling generator (see
                :meth:`compute_confidence_interval`)

        Returns:
            Dict mapping "pass@k" to {"value": ..., "lower": ..., "upper": ...}
        """
        result = {}

        for k in self.k_values:
            value = pass_at_k_estimator(results_per_task, k)
            lower, upper = self.compute_confidence_interval(
                results_per_task, k, confidence, n_bootstrap, seed=seed
            )
            result[f"pass@{k}"] = {
                "value": value,
                "lower": lower,
                "upper": upper,
            }

        return result
