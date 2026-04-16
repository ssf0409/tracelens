"""Tests for statistical inference module."""

import numpy as np
import pytest

from eval_kit.statistics.inference import (
    ComparisonResult,
    MetricEstimate,
    bootstrap_ci,
    bootstrap_difference_ci,
    cohens_d,
    compare_metrics,
    compare_to_baseline_summary,
    estimate_metric,
    permutation_test,
)


class TestMetricEstimate:
    """Tests for MetricEstimate dataclass."""

    def test_creation(self):
        """Test creating a MetricEstimate."""
        est = MetricEstimate(
            mean=0.85,
            std=0.05,
            n=100,
            ci_lower=0.84,
            ci_upper=0.86,
        )
        assert est.mean == 0.85
        assert est.std == 0.05
        assert est.n == 100

    def test_standard_error(self):
        """Test standard error calculation."""
        est = MetricEstimate(mean=0.85, std=0.10, n=100)
        assert est.se == pytest.approx(0.01, rel=0.01)

    def test_ci_width(self):
        """Test CI width calculation."""
        est = MetricEstimate(
            mean=0.85,
            std=0.05,
            n=100,
            ci_lower=0.84,
            ci_upper=0.86,
        )
        assert est.ci_width == pytest.approx(0.02, rel=0.01)

    def test_to_dict(self):
        """Test dictionary conversion."""
        est = MetricEstimate(mean=0.85, std=0.05, n=100)
        d = est.to_dict()
        assert d["mean"] == 0.85
        assert d["std"] == 0.05
        assert d["n"] == 100


class TestBootstrapCI:
    """Tests for bootstrap_ci function."""

    def test_basic_ci(self):
        """Test basic CI calculation."""
        np.random.seed(42)
        values = np.random.normal(0.8, 0.1, 100)

        point, lower, upper = bootstrap_ci(values, confidence=0.95, seed=42)

        assert point == pytest.approx(np.mean(values), rel=0.01)
        assert lower < point < upper
        assert upper - lower < 0.1  # Reasonable CI width

    def test_empty_values(self):
        """Test with empty values."""
        point, lower, upper = bootstrap_ci([])
        assert point == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_single_value(self):
        """Test with single value."""
        point, lower, upper = bootstrap_ci([0.5])
        assert point == 0.5
        assert lower == 0.5
        assert upper == 0.5

    def test_median_statistic(self):
        """Test with median statistic."""
        values = [1, 2, 3, 4, 100]  # Skewed distribution
        point, _, _ = bootstrap_ci(values, statistic="median", seed=42)
        assert point == pytest.approx(np.median(values), rel=0.01)


class TestEstimateMetric:
    """Tests for estimate_metric function."""

    def test_basic_estimate(self):
        """Test basic metric estimation."""
        np.random.seed(42)
        values = np.random.normal(0.85, 0.05, 50)

        est = estimate_metric(values, seed=42)

        assert est.mean == pytest.approx(np.mean(values), rel=0.01)
        assert est.std == pytest.approx(np.std(values, ddof=1), rel=0.1)
        assert est.n == 50
        assert est.ci_lower is not None
        assert est.ci_upper is not None
        assert est.ci_lower < est.mean < est.ci_upper

    def test_empty_values(self):
        """Test with empty values."""
        est = estimate_metric([])
        assert est.mean == 0.0
        assert est.n == 0


class TestBootstrapDifferenceCI:
    """Tests for bootstrap_difference_ci function."""

    def test_significant_difference(self):
        """Test with significantly different distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0.7, 0.05, 50)
        current = np.random.normal(0.85, 0.05, 50)

        delta, lower, upper = bootstrap_difference_ci(
            baseline, current, seed=42
        )

        assert delta > 0  # Current is higher
        assert lower > 0  # CI doesn't include 0 (significant)

    def test_no_difference(self):
        """Test with same distribution."""
        np.random.seed(42)
        values = np.random.normal(0.8, 0.05, 50)

        delta, lower, upper = bootstrap_difference_ci(
            values, values, seed=42
        )

        # CI should include 0
        assert lower <= 0 <= upper


class TestCohensD:
    """Tests for cohens_d function."""

    def test_large_effect(self):
        """Test with large effect size."""
        baseline = [1, 2, 3, 4, 5]
        current = [5, 6, 7, 8, 9]

        d = cohens_d(baseline, current)

        assert d > 0.8  # Large effect

    def test_no_effect(self):
        """Test with no effect."""
        values = [1, 2, 3, 4, 5]

        d = cohens_d(values, values)

        assert d == pytest.approx(0.0, abs=0.01)

    def test_negative_effect(self):
        """Test with negative effect (current < baseline)."""
        baseline = [5, 6, 7, 8, 9]
        current = [1, 2, 3, 4, 5]

        d = cohens_d(baseline, current)

        assert d < -0.8  # Large negative effect


class TestPermutationTest:
    """Tests for permutation_test function."""

    def test_significant_difference(self):
        """Test with significantly different distributions."""
        np.random.seed(42)
        baseline = np.random.normal(0.5, 0.1, 30)
        current = np.random.normal(0.8, 0.1, 30)

        p_value = permutation_test(baseline, current, seed=42)

        assert p_value < 0.05  # Significant

    def test_no_difference(self):
        """Test with same distribution."""
        np.random.seed(42)
        values = np.random.normal(0.8, 0.1, 30)

        p_value = permutation_test(values, values, seed=42)

        # Should not be significant
        assert p_value > 0.05


class TestCompareMetrics:
    """Tests for compare_metrics function."""

    def test_significant_improvement(self):
        """Test detecting significant improvement."""
        np.random.seed(42)
        baseline = np.random.normal(0.7, 0.05, 30)
        current = np.random.normal(0.85, 0.05, 30)

        result = compare_metrics(baseline, current, seed=42)

        assert result.is_significant is True
        assert result.is_improvement is True
        assert result.is_regression is False
        assert result.delta > 0

    def test_significant_regression(self):
        """Test detecting significant regression."""
        np.random.seed(42)
        baseline = np.random.normal(0.85, 0.05, 30)
        current = np.random.normal(0.7, 0.05, 30)

        result = compare_metrics(baseline, current, seed=42)

        assert result.is_significant is True
        assert result.is_regression is True
        assert result.is_improvement is False
        assert result.delta < 0

    def test_no_significant_change(self):
        """Test when change is not significant."""
        np.random.seed(42)
        baseline = np.random.normal(0.8, 0.1, 20)
        current = np.random.normal(0.82, 0.1, 20)  # Small difference

        result = compare_metrics(baseline, current, seed=42)

        # May or may not be significant with small sample, high variance
        # Just verify structure
        assert result.baseline.n == 20
        assert result.current.n == 20

    def test_effect_magnitude(self):
        """Test effect magnitude classification."""
        np.random.seed(42)
        baseline = np.random.normal(0.5, 0.1, 50)
        current = np.random.normal(0.9, 0.1, 50)  # Large difference

        result = compare_metrics(baseline, current, seed=42)

        assert result.effect_magnitude == "large"

    def test_with_p_value(self):
        """Test computing p-value."""
        np.random.seed(42)
        baseline = np.random.normal(0.7, 0.05, 30)
        current = np.random.normal(0.85, 0.05, 30)

        result = compare_metrics(
            baseline, current, compute_p_value=True, seed=42
        )

        assert result.p_value is not None
        assert result.p_value < 0.05

    def test_to_dict(self):
        """Test serialization to dict."""
        np.random.seed(42)
        baseline = np.random.normal(0.8, 0.05, 30)
        current = np.random.normal(0.85, 0.05, 30)

        result = compare_metrics(baseline, current, seed=42)
        d = result.to_dict()

        assert "baseline" in d
        assert "current" in d
        assert "delta" in d
        assert "is_significant" in d
        assert "cohens_d" in d


class TestCompareToBaselineSummary:
    """Tests for compare_to_baseline_summary function."""

    def test_significant_difference(self):
        """Test with significant difference using summary stats."""
        result = compare_to_baseline_summary(
            baseline_mean=0.7,
            baseline_std=0.05,
            baseline_n=100,
            current_mean=0.85,
            current_std=0.05,
            current_n=100,
        )

        assert result.is_significant is True
        assert result.is_improvement is True
        assert result.delta == pytest.approx(0.15, rel=0.01)
        assert result.p_value < 0.05

    def test_no_significant_difference(self):
        """Test with no significant difference."""
        result = compare_to_baseline_summary(
            baseline_mean=0.8,
            baseline_std=0.1,
            baseline_n=20,
            current_mean=0.82,
            current_std=0.1,
            current_n=20,
        )

        # Small difference with high variance and small n
        # CI should include 0
        assert result.ci_lower < 0 < result.ci_upper

    def test_regression(self):
        """Test detecting regression."""
        result = compare_to_baseline_summary(
            baseline_mean=0.85,
            baseline_std=0.05,
            baseline_n=100,
            current_mean=0.7,
            current_std=0.05,
            current_n=100,
        )

        assert result.is_significant is True
        assert result.is_regression is True
        assert result.delta < 0


class TestComparisonResult:
    """Tests for ComparisonResult dataclass."""

    def test_effect_magnitude_negligible(self):
        """Test negligible effect classification."""
        result = ComparisonResult(
            baseline=MetricEstimate(mean=0.8, std=0.1, n=50),
            current=MetricEstimate(mean=0.81, std=0.1, n=50),
            delta=0.01,
            relative_delta=0.0125,
            ci_lower=-0.02,
            ci_upper=0.04,
            confidence=0.95,
            is_significant=False,
            cohens_d=0.1,
            p_value=0.5,
        )
        assert result.effect_magnitude == "negligible"

    def test_effect_magnitude_small(self):
        """Test small effect classification."""
        result = ComparisonResult(
            baseline=MetricEstimate(mean=0.8, std=0.1, n=50),
            current=MetricEstimate(mean=0.83, std=0.1, n=50),
            delta=0.03,
            relative_delta=0.0375,
            ci_lower=0.01,
            ci_upper=0.05,
            confidence=0.95,
            is_significant=True,
            cohens_d=0.3,
            p_value=0.03,
        )
        assert result.effect_magnitude == "small"

    def test_effect_magnitude_medium(self):
        """Test medium effect classification."""
        result = ComparisonResult(
            baseline=MetricEstimate(mean=0.8, std=0.1, n=50),
            current=MetricEstimate(mean=0.87, std=0.1, n=50),
            delta=0.07,
            relative_delta=0.0875,
            ci_lower=0.04,
            ci_upper=0.10,
            confidence=0.95,
            is_significant=True,
            cohens_d=0.6,
            p_value=0.001,
        )
        assert result.effect_magnitude == "medium"

    def test_effect_magnitude_large(self):
        """Test large effect classification."""
        result = ComparisonResult(
            baseline=MetricEstimate(mean=0.5, std=0.1, n=50),
            current=MetricEstimate(mean=0.7, std=0.1, n=50),
            delta=0.2,
            relative_delta=0.4,
            ci_lower=0.15,
            ci_upper=0.25,
            confidence=0.95,
            is_significant=True,
            cohens_d=1.5,
            p_value=0.0001,
        )
        assert result.effect_magnitude == "large"
