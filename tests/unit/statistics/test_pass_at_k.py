"""Tests for pass@k statistics."""

import pytest

from agent_eval.statistics.pass_at_k import (
    pass_at_k,
    pass_at_k_estimator,
    PassAtKAnalyzer,
)


class TestPassAtK:
    """Tests for pass_at_k function."""

    def test_pass_at_k_all_pass(self):
        """Test when all samples pass."""
        # 10 samples, all correct, k=5
        result = pass_at_k(n=10, c=10, k=5)
        assert result == 1.0

    def test_pass_at_k_none_pass(self):
        """Test when no samples pass."""
        # 10 samples, none correct, k=5
        result = pass_at_k(n=10, c=0, k=5)
        assert result == 0.0

    def test_pass_at_k_half_pass(self):
        """Test when half samples pass."""
        # 10 samples, 5 correct, k=3
        result = pass_at_k(n=10, c=5, k=3)
        # Should be high (likely at least 1 of 3 passes)
        assert result > 0.8

    def test_pass_at_k_single_pass(self):
        """Test when only one sample passes."""
        # 10 samples, 1 correct, k=5
        result = pass_at_k(n=10, c=1, k=5)
        # 50% chance of picking the one that passes
        assert result == pytest.approx(0.5, rel=0.01)

    def test_pass_at_k_k_equals_n(self):
        """Test when k equals n."""
        # 5 samples, 3 correct, k=5
        result = pass_at_k(n=5, c=3, k=5)
        # Must include at least one of the 3 correct
        assert result == 1.0

    def test_pass_at_k_k_greater_than_failures(self):
        """Test when k is greater than number of failures."""
        # 5 samples, 4 correct, k=3 (only 1 failure)
        result = pass_at_k(n=5, c=4, k=3)
        assert result == 1.0


class TestPassAtKEstimator:
    """Tests for pass_at_k_estimator function."""

    def test_empty_results(self):
        """Test with empty results."""
        result = pass_at_k_estimator({}, k=3)
        assert result == 0.0

    def test_single_task_all_pass(self):
        """Test single task with all passes."""
        results = {"task1": [True] * 10}
        result = pass_at_k_estimator(results, k=5)
        assert result == 1.0

    def test_single_task_mixed(self):
        """Test single task with mixed results."""
        results = {"task1": [True, True, False, True, True, False, True, True, False, True]}
        result = pass_at_k_estimator(results, k=3)
        # 7/10 pass, high probability at least 1 of 3 passes
        assert result > 0.9

    def test_multiple_tasks(self, pass_results_by_task):
        """Test with multiple tasks."""
        result = pass_at_k_estimator(pass_results_by_task, k=3)
        # Average across tasks
        assert 0.5 < result < 1.0

    def test_insufficient_samples(self):
        """Test with fewer samples than k."""
        results = {"task1": [True, False]}  # Only 2 samples, k=5
        result = pass_at_k_estimator(results, k=5)
        # Falls back to empirical estimate: 0.5
        assert result == 0.5


class TestPassAtKAnalyzer:
    """Tests for PassAtKAnalyzer class."""

    def test_default_k_values(self):
        """Test default k values."""
        analyzer = PassAtKAnalyzer()
        assert analyzer.k_values == [1, 5, 10]

    def test_custom_k_values(self):
        """Test custom k values."""
        analyzer = PassAtKAnalyzer(k_values=[1, 3, 5])
        assert analyzer.k_values == [1, 3, 5]

    def test_analyze(self, pass_results_by_task):
        """Test analyze method."""
        analyzer = PassAtKAnalyzer(k_values=[1, 3, 5])
        results = analyzer.analyze(pass_results_by_task)

        assert "pass@1" in results
        assert "pass@3" in results
        assert "pass@5" in results

        # pass@k should increase with k
        assert results["pass@1"] <= results["pass@3"] <= results["pass@5"]

    def test_confidence_interval(self, pass_results_by_task):
        """Test confidence interval calculation."""
        analyzer = PassAtKAnalyzer()
        lower, upper = analyzer.compute_confidence_interval(
            pass_results_by_task, k=3, confidence=0.95, n_bootstrap=100
        )

        assert lower <= upper
        assert 0 <= lower <= 1
        assert 0 <= upper <= 1

    def test_analyze_with_ci(self, pass_results_by_task):
        """Test analyze with confidence intervals."""
        analyzer = PassAtKAnalyzer(k_values=[1, 3])
        results = analyzer.analyze_with_ci(
            pass_results_by_task, confidence=0.95, n_bootstrap=100
        )

        assert "pass@1" in results
        assert "value" in results["pass@1"]
        assert "lower" in results["pass@1"]
        assert "upper" in results["pass@1"]
