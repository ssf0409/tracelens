"""pass^k (consistency) metric implementation.

pass^k measures the probability that ALL of k consecutive samples pass.
Higher values indicate more reliable/consistent performance.

This is a reliability metric - it answers "is the agent consistent?"
Higher k values require more consecutive successes, so pass^k decreases with k.
"""

import numpy as np


def pass_to_k(results: list[bool], k: int) -> float:
    """Calculate pass^k (consistency) metric.

    Measures the proportion of k-length windows where all samples pass.
    A sliding window approach is used.

    Args:
        results: List of pass/fail booleans from multiple runs
        k: Number of consecutive passes required

    Returns:
        Proportion of k-length windows where all samples pass (0.0 to 1.0)

    Example:
        >>> pass_to_k([True, True, True, True, True], 3)
        1.0  # All windows of 3 pass

        >>> pass_to_k([True, True, False, True, True], 3)
        0.333...  # Only 1 of 3 windows passes
    """
    if len(results) < k:
        return 0.0

    n_windows = len(results) - k + 1
    consistent_windows = 0

    for i in range(n_windows):
        if all(results[i:i + k]):
            consistent_windows += 1

    return consistent_windows / n_windows


def pass_to_k_estimator(
    results_per_task: dict[str, list[bool]],
    k: int,
) -> float:
    """Compute pass^k (consistency) across multiple tasks.

    For each task with enough samples, computes pass^k.
    Returns the average pass^k across all eligible tasks.

    Args:
        results_per_task: Dict mapping task_id to list of pass/fail booleans
        k: Number of consecutive passes required

    Returns:
        Average pass^k across all tasks with >= k samples

    Example:
        >>> results = {
        ...     "task1": [True, True, True, True, True],
        ...     "task2": [True, True, False, True, True],
        ... }
        >>> pass_to_k_estimator(results, k=3)
        0.666...  # task1: 1.0, task2: 0.333
    """
    if not results_per_task:
        return 0.0

    scores = []

    for task_id, results in results_per_task.items():
        if len(results) >= k:
            scores.append(pass_to_k(results, k))

    return float(np.mean(scores)) if scores else 0.0


class ConsistencyAnalyzer:
    """Analyzer for pass^k consistency metrics.

    Computes pass^k for multiple k values and provides reliability scoring.

    Example:
        analyzer = ConsistencyAnalyzer(k_values=[2, 3, 5])
        results = analyzer.analyze(pass_results_by_task)
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
        results_per_task: dict[str, list[bool]],
    ) -> dict[str, float]:
        """Compute pass^k for multiple k values.

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans

        Returns:
            Dict mapping "pass^k" to computed value
        """
        return {
            f"pass^{k}": pass_to_k_estimator(results_per_task, k)
            for k in self.k_values
        }

    def compute_reliability_score(
        self,
        results_per_task: dict[str, list[bool]],
    ) -> float:
        """Compute overall reliability score.

        Combines pass^k metrics weighted by k to give higher weight
        to longer consistent runs. A higher score indicates more
        reliable/consistent performance.

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans

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
        results_per_task: dict[str, list[bool]],
    ) -> dict[str, float]:
        """Compute additional stability metrics.

        Returns:
            Dict with:
            - "pass^k" values for each k
            - "reliability_score": weighted combination
            - "failure_rate": proportion of failed trials
            - "longest_streak": average longest passing streak per task
        """
        metrics = self.analyze(results_per_task)
        metrics["reliability_score"] = self.compute_reliability_score(results_per_task)

        # Compute failure rate
        all_results = []
        for results in results_per_task.values():
            all_results.extend(results)

        if all_results:
            metrics["failure_rate"] = 1.0 - (sum(all_results) / len(all_results))
        else:
            metrics["failure_rate"] = 1.0

        # Compute average longest streak
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
