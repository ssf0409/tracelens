"""pass@k metric implementation.

pass@k estimates the probability that at least one of k samples passes,
given n total samples with c correct.

This is a capability metric - it answers "can the agent do this at all?"
Higher k values give the agent more chances, so pass@k increases with k.

Reference: Chen et al., "Evaluating Large Language Models Trained on Code"
"""

import numpy as np


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


def pass_at_k_estimator(
    results_per_task: dict[str, list[bool]],
    k: int,
) -> float:
    """Compute pass@k across multiple tasks.

    For each task, computes pass@k using available samples.
    Returns the average pass@k across all tasks.

    Args:
        results_per_task: Dict mapping task_id to list of pass/fail booleans
        k: Number of samples to consider

    Returns:
        Average pass@k across all tasks

    Example:
        >>> results = {
        ...     "task1": [True, True, False, True, True],
        ...     "task2": [False, True, False, False, True],
        ... }
        >>> pass_at_k_estimator(results, k=3)
        0.9...  # High probability task1 passes, lower for task2
    """
    if not results_per_task:
        return 0.0

    scores = []

    for task_id, results in results_per_task.items():
        n = len(results)
        c = sum(results)

        if n >= k:
            scores.append(pass_at_k(n, c, k))
        elif n > 0:
            # Not enough samples, use empirical estimate
            scores.append(c / n if n > 0 else 0.0)

    return float(np.mean(scores)) if scores else 0.0


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

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans

        Returns:
            Dict mapping "pass@k" to computed value
        """
        return {
            f"pass@{k}": pass_at_k_estimator(results_per_task, k)
            for k in self.k_values
        }

    def compute_confidence_interval(
        self,
        results_per_task: dict[str, list[bool]],
        k: int,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for pass@k.

        Uses bootstrap resampling over tasks to estimate the confidence
        interval for pass@k.

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans
            k: k value for pass@k
            confidence: Confidence level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not results_per_task:
            return 0.0, 0.0

        task_ids = list(results_per_task.keys())
        bootstrap_scores = []

        rng = np.random.default_rng()

        for _ in range(n_bootstrap):
            # Bootstrap resample tasks
            sampled_ids = rng.choice(task_ids, size=len(task_ids), replace=True)
            sampled_results = {tid: results_per_task[tid] for tid in sampled_ids}
            bootstrap_scores.append(pass_at_k_estimator(sampled_results, k))

        alpha = (1 - confidence) / 2
        lower = float(np.percentile(bootstrap_scores, alpha * 100))
        upper = float(np.percentile(bootstrap_scores, (1 - alpha) * 100))

        return lower, upper

    def analyze_with_ci(
        self,
        results_per_task: dict[str, list[bool]],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> dict[str, dict[str, float]]:
        """Compute pass@k with confidence intervals.

        Args:
            results_per_task: Dict mapping task_id to list of pass/fail booleans
            confidence: Confidence level (default 0.95)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dict mapping "pass@k" to {"value": ..., "lower": ..., "upper": ...}
        """
        result = {}

        for k in self.k_values:
            value = pass_at_k_estimator(results_per_task, k)
            lower, upper = self.compute_confidence_interval(
                results_per_task, k, confidence, n_bootstrap
            )
            result[f"pass@{k}"] = {
                "value": value,
                "lower": lower,
                "upper": upper,
            }

        return result
