"""Statistical inference for metric comparison.

Provides bootstrap-based confidence intervals and significance testing
for comparing evaluation metrics against baselines.

This module implements research-grade statistical rigor:
- Bootstrap confidence intervals for metric estimates
- Statistical significance testing for regression detection
- Effect size calculations (Cohen's d)
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class MetricEstimate:
    """A metric estimate with uncertainty quantification.

    Stores the point estimate along with confidence bounds
    and sample statistics for research-grade reporting.
    """

    mean: float
    std: float
    n: int
    ci_lower: float | None = None
    ci_upper: float | None = None
    confidence: float = 0.95

    @property
    def se(self) -> float:
        """Standard error of the mean."""
        if self.n <= 1:
            return float("inf")
        return float(self.std / np.sqrt(self.n))

    @property
    def ci_width(self) -> float | None:
        """Width of the confidence interval."""
        if self.ci_lower is None or self.ci_upper is None:
            return None
        return self.ci_upper - self.ci_lower

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mean": self.mean,
            "std": self.std,
            "n": self.n,
            "se": self.se,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        if self.ci_lower is not None:
            return f"MetricEstimate({self.mean:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}], n={self.n})"
        return f"MetricEstimate({self.mean:.4f} ± {self.std:.4f}, n={self.n})"


@dataclass
class ComparisonResult:
    """Result of comparing two metric estimates.

    Contains statistical test results for determining if
    the difference is significant.
    """

    baseline: MetricEstimate
    current: MetricEstimate
    delta: float  # current.mean - baseline.mean
    relative_delta: float  # (current - baseline) / |baseline|
    ci_lower: float | None  # CI of the difference
    ci_upper: float | None
    confidence: float
    is_significant: bool  # CI doesn't include 0
    cohens_d: float | None  # Effect size
    p_value: float | None  # From permutation test (if computed)

    @property
    def is_regression(self) -> bool:
        """Check if this represents a statistically significant regression.

        A regression occurs when the difference is significantly negative
        (for higher-is-better metrics).
        """
        if not self.is_significant:
            return False
        return self.delta < 0

    @property
    def is_improvement(self) -> bool:
        """Check if this represents a statistically significant improvement."""
        if not self.is_significant:
            return False
        return self.delta > 0

    @property
    def effect_magnitude(self) -> str:
        """Classify effect size magnitude (Cohen's conventions)."""
        if self.cohens_d is None:
            return "unknown"
        d = abs(self.cohens_d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "baseline": self.baseline.to_dict(),
            "current": self.current.to_dict(),
            "delta": self.delta,
            "relative_delta": self.relative_delta,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "confidence": self.confidence,
            "is_significant": self.is_significant,
            "is_regression": self.is_regression,
            "is_improvement": self.is_improvement,
            "cohens_d": self.cohens_d,
            "effect_magnitude": self.effect_magnitude,
            "p_value": self.p_value,
        }


def bootstrap_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Uses percentile bootstrap method for simplicity and robustness.

    Args:
        values: Sample values
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples
        statistic: "mean", "median", or "std"
        seed: Random seed for reproducibility

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)
    """
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    if n == 1:
        return float(values[0]), float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)

    # Compute bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats.append(np.median(sample))
        elif statistic == "std":
            bootstrap_stats.append(np.std(sample, ddof=1))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    # Compute percentile CI
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstrap_stats, alpha * 100))
    upper = float(np.percentile(bootstrap_stats, (1 - alpha) * 100))

    # Point estimate
    if statistic == "mean":
        point = float(np.mean(values))
    elif statistic == "median":
        point = float(np.median(values))
    else:
        point = float(np.std(values, ddof=1))

    return point, lower, upper


def estimate_metric(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> MetricEstimate:
    """Estimate a metric with confidence interval.

    Args:
        values: Sample values
        confidence: Confidence level
        n_bootstrap: Bootstrap samples
        seed: Random seed

    Returns:
        MetricEstimate with mean, std, n, and CI
    """
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return MetricEstimate(
            mean=0.0, std=0.0, n=0,
            ci_lower=0.0, ci_upper=0.0, confidence=confidence
        )

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    # Bootstrap CI for the mean
    _, ci_lower, ci_upper = bootstrap_ci(
        values, confidence, n_bootstrap, "mean", seed
    )

    return MetricEstimate(
        mean=mean,
        std=std,
        n=n,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
    )


def bootstrap_difference_ci(
    baseline_values: list[float] | np.ndarray,
    current_values: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap CI for the difference in means.

    Uses independent bootstrap sampling from both distributions.

    Args:
        baseline_values: Baseline sample values
        current_values: Current sample values
        confidence: Confidence level
        n_bootstrap: Bootstrap samples
        seed: Random seed

    Returns:
        Tuple of (point_estimate, lower, upper) for (current - baseline)
    """
    baseline = np.asarray(baseline_values)
    current = np.asarray(current_values)

    if len(baseline) == 0 or len(current) == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)

    # Bootstrap distribution of differences
    diffs = []
    for _ in range(n_bootstrap):
        b_sample = rng.choice(baseline, size=len(baseline), replace=True)
        c_sample = rng.choice(current, size=len(current), replace=True)
        diffs.append(np.mean(c_sample) - np.mean(b_sample))

    # Percentile CI
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(diffs, alpha * 100))
    upper = float(np.percentile(diffs, (1 - alpha) * 100))

    point = float(np.mean(current) - np.mean(baseline))

    return point, lower, upper


def cohens_d(
    baseline_values: list[float] | np.ndarray,
    current_values: list[float] | np.ndarray,
) -> float:
    """Compute Cohen's d effect size.

    Uses pooled standard deviation for independent samples.

    Args:
        baseline_values: Baseline sample
        current_values: Current sample

    Returns:
        Cohen's d (positive = current > baseline)
    """
    baseline = np.asarray(baseline_values)
    current = np.asarray(current_values)

    n1, n2 = len(baseline), len(current)

    if n1 < 2 or n2 < 2:
        return 0.0

    mean_diff = np.mean(current) - np.mean(baseline)

    # Pooled standard deviation
    var1 = np.var(baseline, ddof=1)
    var2 = np.var(current, ddof=1)
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return float(mean_diff / pooled_std)


def permutation_test(
    baseline_values: list[float] | np.ndarray,
    current_values: list[float] | np.ndarray,
    n_permutations: int = 10000,
    alternative: str = "two-sided",
    seed: int | None = None,
) -> float:
    """Compute p-value using permutation test.

    Tests the null hypothesis that baseline and current come from
    the same distribution.

    Args:
        baseline_values: Baseline sample
        current_values: Current sample
        n_permutations: Number of permutations
        alternative: "two-sided", "greater", or "less"
        seed: Random seed

    Returns:
        p-value
    """
    baseline = np.asarray(baseline_values)
    current = np.asarray(current_values)

    if len(baseline) == 0 or len(current) == 0:
        return 1.0

    rng = np.random.default_rng(seed)

    # Observed difference
    observed = np.mean(current) - np.mean(baseline)

    # Pool the data
    combined = np.concatenate([baseline, current])
    n_baseline = len(baseline)

    # Permutation distribution
    perm_diffs_list: list[float] = []
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_baseline = combined[:n_baseline]
        perm_current = combined[n_baseline:]
        perm_diffs_list.append(float(np.mean(perm_current) - np.mean(perm_baseline)))

    perm_diffs = np.array(perm_diffs_list)

    # Compute p-value
    if alternative == "two-sided":
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed))
    elif alternative == "greater":
        p_value = np.mean(perm_diffs >= observed)
    elif alternative == "less":
        p_value = np.mean(perm_diffs <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    return float(p_value)


def compare_metrics(
    baseline_values: list[float] | np.ndarray,
    current_values: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    compute_p_value: bool = False,
    seed: int | None = None,
) -> ComparisonResult:
    """Compare current metrics against baseline with statistical inference.

    Computes bootstrap CI for the difference and determines
    statistical significance.

    Args:
        baseline_values: Baseline sample values
        current_values: Current sample values
        confidence: Confidence level for CI
        n_bootstrap: Bootstrap samples
        compute_p_value: Whether to compute permutation p-value
        seed: Random seed

    Returns:
        ComparisonResult with full statistical analysis
    """
    baseline_est = estimate_metric(baseline_values, confidence, n_bootstrap, seed)
    current_est = estimate_metric(current_values, confidence, n_bootstrap, seed)

    delta, ci_lower, ci_upper = bootstrap_difference_ci(
        baseline_values, current_values, confidence, n_bootstrap, seed
    )

    # Relative delta (handle zero baseline)
    if baseline_est.mean != 0:
        relative_delta = delta / abs(baseline_est.mean)
    else:
        relative_delta = float("inf") if delta != 0 else 0.0

    # Statistical significance: CI doesn't include 0
    is_significant = not (ci_lower <= 0 <= ci_upper)

    # Effect size
    d = cohens_d(baseline_values, current_values)

    # Optional p-value
    p_value = None
    if compute_p_value:
        p_value = permutation_test(
            baseline_values, current_values,
            n_permutations=n_bootstrap,
            seed=seed
        )

    return ComparisonResult(
        baseline=baseline_est,
        current=current_est,
        delta=delta,
        relative_delta=relative_delta,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence=confidence,
        is_significant=is_significant,
        cohens_d=d,
        p_value=p_value,
    )


def compare_to_baseline_summary(
    baseline_mean: float,
    baseline_std: float,
    baseline_n: int,
    current_mean: float,
    current_std: float,
    current_n: int,
    confidence: float = 0.95,
) -> ComparisonResult:
    """Compare metrics when only summary statistics are available.

    Uses Welch's t-test approximation for the CI when raw data
    is not available.

    Args:
        baseline_mean: Baseline mean
        baseline_std: Baseline std deviation
        baseline_n: Baseline sample size
        current_mean: Current mean
        current_std: Current std deviation
        current_n: Current sample size
        confidence: Confidence level

    Returns:
        ComparisonResult (note: effect size may be less accurate)
    """
    baseline_est = MetricEstimate(
        mean=baseline_mean,
        std=baseline_std,
        n=baseline_n,
        confidence=confidence,
    )
    current_est = MetricEstimate(
        mean=current_mean,
        std=current_std,
        n=current_n,
        confidence=confidence,
    )

    delta = current_mean - baseline_mean

    # Relative delta
    if baseline_mean != 0:
        relative_delta = delta / abs(baseline_mean)
    else:
        relative_delta = float("inf") if delta != 0 else 0.0

    # Welch's t-test CI for difference
    se_diff = np.sqrt(
        (baseline_std ** 2 / baseline_n) +
        (current_std ** 2 / current_n)
    )

    # Degrees of freedom (Welch-Satterthwaite)
    if baseline_std > 0 and current_std > 0:
        num = (baseline_std ** 2 / baseline_n + current_std ** 2 / current_n) ** 2
        denom = (
            (baseline_std ** 2 / baseline_n) ** 2 / (baseline_n - 1) +
            (current_std ** 2 / current_n) ** 2 / (current_n - 1)
        )
        df = num / denom if denom > 0 else 1
    else:
        df = baseline_n + current_n - 2

    # CI using t-distribution
    alpha = (1 - confidence) / 2
    t_crit = stats.t.ppf(1 - alpha, df)

    ci_lower = delta - t_crit * se_diff
    ci_upper = delta + t_crit * se_diff

    # Significance
    is_significant = not (ci_lower <= 0 <= ci_upper)

    # Effect size (approximate)
    pooled_std = np.sqrt(
        ((baseline_n - 1) * baseline_std ** 2 +
         (current_n - 1) * current_std ** 2) /
        (baseline_n + current_n - 2)
    ) if baseline_std > 0 or current_std > 0 else 1.0

    cohens_d_val = delta / pooled_std if pooled_std > 0 else 0.0

    # P-value from t-test
    t_stat = delta / se_diff if se_diff > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t_stat), df))

    return ComparisonResult(
        baseline=baseline_est,
        current=current_est,
        delta=delta,
        relative_delta=relative_delta,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence=confidence,
        is_significant=is_significant,
        cohens_d=float(cohens_d_val),
        p_value=p_value,
    )
