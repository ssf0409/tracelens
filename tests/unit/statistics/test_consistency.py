"""Tests for pass^k (consistency) statistics."""

import pytest

from tracelens.statistics.consistency import (
    ConsistencyAnalyzer,
    pass_to_k,
    pass_to_k_estimator,
)


class TestPassToK:
    """Tests for pass_to_k function."""

    def test_all_pass(self):
        """Test when all samples pass."""
        results = [True] * 10
        result = pass_to_k(results, k=3)
        # All windows of 3 pass
        assert result == 1.0

    def test_all_fail(self):
        """Test when all samples fail."""
        results = [False] * 10
        result = pass_to_k(results, k=3)
        assert result == 0.0

    def test_single_failure(self):
        """Test with a single failure in the middle."""
        results = [True, True, True, False, True, True, True]
        result = pass_to_k(results, k=3)
        # Windows: [TTT], [TTF], [TFT], [FTT], [TTT]
        # Passing: 2 out of 5
        assert result == pytest.approx(2 / 5, rel=0.01)

    def test_alternating(self):
        """Test with alternating pass/fail."""
        results = [True, False, True, False, True]
        result = pass_to_k(results, k=2)
        # Windows: [TF], [FT], [TF], [FT]
        # None pass (need 2 consecutive True)
        assert result == 0.0

    def test_insufficient_samples(self):
        """Test with fewer samples than k."""
        results = [True, True]
        result = pass_to_k(results, k=5)
        assert result == 0.0

    def test_k_equals_one(self):
        """Test k=1 (just pass rate)."""
        results = [True, True, False, True, False]
        result = pass_to_k(results, k=1)
        assert result == 3 / 5


class TestPassToKEstimator:
    """Tests for pass_to_k_estimator function."""

    def test_empty_results(self):
        """Test with empty results."""
        result = pass_to_k_estimator({}, k=3)
        assert result == 0.0

    def test_single_task(self):
        """Test single task."""
        results = {"task1": [True, True, True, False, True]}
        result = pass_to_k_estimator(results, k=3)
        # Windows: [TTT], [TTF], [TFT]
        # Only 1 passes
        assert result == pytest.approx(1 / 3, rel=0.01)

    def test_multiple_tasks(self, pass_results_by_task):
        """Test with multiple tasks."""
        result = pass_to_k_estimator(pass_results_by_task, k=3)
        assert 0 <= result <= 1

    def test_skip_insufficient_samples(self):
        """Test that tasks with insufficient samples are skipped."""
        results = {
            "task1": [True, True, True, True, True],  # 5 samples
            "task2": [True, True],  # Only 2 samples, skipped for k=3
        }
        result = pass_to_k_estimator(results, k=3)
        # Only task1 is counted
        assert result == 1.0


class TestConsistencyAnalyzer:
    """Tests for ConsistencyAnalyzer class."""

    def test_default_k_values(self):
        """Test default k values."""
        analyzer = ConsistencyAnalyzer()
        assert analyzer.k_values == [2, 3, 5]

    def test_custom_k_values(self):
        """Test custom k values."""
        analyzer = ConsistencyAnalyzer(k_values=[2, 4, 8])
        assert analyzer.k_values == [2, 4, 8]

    def test_analyze(self, pass_results_by_task):
        """Test analyze method."""
        analyzer = ConsistencyAnalyzer(k_values=[2, 3, 5])
        results = analyzer.analyze(pass_results_by_task)

        assert "pass^2" in results
        assert "pass^3" in results
        assert "pass^5" in results

        # pass^k should decrease with k (harder to pass all)
        assert results["pass^2"] >= results["pass^3"] >= results["pass^5"]

    def test_compute_reliability_score(self, pass_results_by_task):
        """Test reliability score computation."""
        analyzer = ConsistencyAnalyzer(k_values=[2, 3, 5])
        score = analyzer.compute_reliability_score(pass_results_by_task)

        assert 0 <= score <= 1

    def test_compute_stability_metrics(self, pass_results_by_task):
        """Test stability metrics computation."""
        analyzer = ConsistencyAnalyzer(k_values=[2, 3])
        metrics = analyzer.compute_stability_metrics(pass_results_by_task)

        assert "pass^2" in metrics
        assert "pass^3" in metrics
        assert "reliability_score" in metrics
        assert "failure_rate" in metrics
        assert "avg_longest_streak" in metrics

    def test_empty_results(self):
        """Test with empty results."""
        analyzer = ConsistencyAnalyzer()

        result = analyzer.analyze({})
        assert all(v == 0.0 for v in result.values())

        score = analyzer.compute_reliability_score({})
        assert score == 0.0


class TestPassToKOrderAndGaps:
    """Issue #45: consecutive windows follow run order and never span a gap."""

    def test_corrected_docstring_example(self):
        # Windows TTF, TFT, FTT: none is all-pass. (The old docstring claimed 1/3.)
        assert pass_to_k([True, True, False, True, True], 3) == 0.0
        assert pass_to_k([True, True, False, True, True], 2) == 0.5

    def test_order_changes_the_statistic(self):
        assert pass_to_k([True, True, False, False], 2) == pytest.approx(1 / 3)
        assert pass_to_k([True, False, False, True], 2) == 0.0

    def test_gap_breaks_windows(self):
        # Windows (T, None) and (None, T) are incomplete; TT and TT remain.
        assert pass_to_k([True, True, None, True, True], 2) == 1.0
        # Only one complete window, and it fails.
        assert pass_to_k([False, None, True, False], 2) == 0.0
        # No complete window at all.
        assert pass_to_k([True, None, True], 2) == 0.0

    def test_estimator_drops_tasks_without_a_complete_window(self):
        results = {
            "gappy": [True, None, True],          # no complete window of 2
            "ok": [True, True, False],            # TT, TF -> 0.5
        }
        assert pass_to_k_estimator(results, k=2) == 0.5

    def test_estimator_corrected_docstring_example(self):
        results = {
            "task1": [True, True, True, True, True],
            "task2": [True, True, False, True, True],
        }
        assert pass_to_k_estimator(results, k=3) == 0.5

    def test_k_below_one_raises(self):
        with pytest.raises(ValueError, match="k must be at least 1"):
            pass_to_k([True, True], 0)
        with pytest.raises(ValueError, match="k must be at least 1"):
            pass_to_k_estimator({"a": [True, True]}, 0)

    def test_stability_metrics_handle_gaps(self):
        analyzer = ConsistencyAnalyzer(k_values=[2])
        metrics = analyzer.compute_stability_metrics(
            {"a": [True, None, True, True, False]}
        )
        # Observed runs T, T, T, F -> failure rate 1/4; the gap breaks the streak -> 2.
        assert metrics["failure_rate"] == pytest.approx(0.25)
        assert metrics["avg_longest_streak"] == 2.0
        # Windows: (T,None) skip, (None,T) skip, (T,T) pass, (T,F) fail -> 0.5
        assert metrics["pass^2"] == 0.5
