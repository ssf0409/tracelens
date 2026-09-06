"""Tests for pass@k statistics."""

import numpy as np
import pytest

from tracelens.statistics.pass_at_k import (
    PassAtKAnalyzer,
    pass_at_k,
    pass_at_k_estimator,
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


class TestBootstrapMultiplicity:
    """Issue #44: bootstrap resamples must preserve repeated task draws."""

    @staticmethod
    def _force_draws(monkeypatch, indices):
        """Make every bootstrap resample draw exactly ``indices``.

        Indices are positions in sorted task_id order, which is the order
        the analyzer computes per-task scores in.
        """

        class _ForcedRng:
            def choice(self, values, size=None, replace=True):
                return np.asarray(values)[np.asarray(indices)]

        monkeypatch.setattr(np.random, "default_rng", lambda seed=None: _ForcedRng())

    def test_draw_a_a_b_counts_a_twice(self, monkeypatch):
        # sorted order: a -> 1.0, b -> 0.0, c -> 1.0
        results = {"a": [True], "b": [False], "c": [True]}
        self._force_draws(monkeypatch, [0, 0, 1])  # [A, A, B]
        lower, upper = PassAtKAnalyzer().compute_confidence_interval(
            results, k=1, n_bootstrap=1
        )
        # Hand-derived: mean of [1, 1, 0] = 2/3. The pre-fix implementation
        # collapsed the draw to {A, B} and reported 1/2.
        assert lower == pytest.approx(2 / 3)
        assert upper == pytest.approx(2 / 3)

    def test_draw_b_b_a_counts_b_twice(self, monkeypatch):
        results = {"a": [True], "b": [False], "c": [True]}
        self._force_draws(monkeypatch, [1, 1, 0])  # [B, B, A]
        lower, upper = PassAtKAnalyzer().compute_confidence_interval(
            results, k=1, n_bootstrap=1
        )
        assert lower == pytest.approx(1 / 3)
        assert upper == pytest.approx(1 / 3)

    def test_all_identical_tasks_give_degenerate_interval(self):
        analyzer = PassAtKAnalyzer()
        all_pass = {f"t{i}": [True, True, True] for i in range(5)}
        all_fail = {f"t{i}": [False, False, False] for i in range(5)}
        assert analyzer.compute_confidence_interval(all_pass, k=2, seed=0) == (1.0, 1.0)
        assert analyzer.compute_confidence_interval(all_fail, k=2, seed=0) == (0.0, 0.0)

    def test_seed_is_reproducible(self, pass_results_by_task):
        analyzer = PassAtKAnalyzer()
        first = analyzer.compute_confidence_interval(pass_results_by_task, k=3, seed=7)
        second = analyzer.compute_confidence_interval(pass_results_by_task, k=3, seed=7)
        assert first == second

    def test_task_order_does_not_change_interval(self, pass_results_by_task):
        analyzer = PassAtKAnalyzer()
        reordered = dict(reversed(list(pass_results_by_task.items())))
        assert list(reordered) != list(pass_results_by_task)
        assert analyzer.compute_confidence_interval(
            pass_results_by_task, k=3, seed=11
        ) == analyzer.compute_confidence_interval(reordered, k=3, seed=11)

    def test_matches_independent_reference_bootstrap(self):
        """Cross-check against a reference resampler that is not the production code."""
        rng = np.random.default_rng(123)
        n_runs, k = 6, 2
        results = {
            f"task{i:02d}": [bool(rng.random() < p) for _ in range(n_runs)]
            for i, p in enumerate(rng.uniform(0.05, 0.95, size=12))
        }
        # Reference per-task scores via the public single-task function only.
        scores = np.array([
            pass_at_k(n_runs, sum(results[tid]), k) for tid in sorted(results)
        ])
        n_boot = 20000
        ref_rng = np.random.default_rng(999)
        idx = ref_rng.integers(0, len(scores), size=(n_boot, len(scores)))
        ref_means = scores[idx].mean(axis=1)
        ref_lower, ref_upper = np.percentile(ref_means, [2.5, 97.5])

        lower, upper = PassAtKAnalyzer().compute_confidence_interval(
            results, k=k, confidence=0.95, n_bootstrap=n_boot, seed=1
        )
        assert lower == pytest.approx(ref_lower, abs=0.02)
        assert upper == pytest.approx(ref_upper, abs=0.02)

        # The pre-fix algorithm (mean over the *distinct* tasks drawn) yields
        # a visibly narrower interval; the fix must not reproduce it.
        dedup_means = np.array([scores[np.unique(row)].mean() for row in idx])
        dedup_lower, dedup_upper = np.percentile(dedup_means, [2.5, 97.5])
        assert (upper - lower) > 1.1 * (dedup_upper - dedup_lower)

    def test_empty_and_single_task_inputs(self):
        analyzer = PassAtKAnalyzer()
        assert analyzer.compute_confidence_interval({}, k=1, seed=0) == (0.0, 0.0)
        single = {"only": [True, False, True]}
        score = pass_at_k(3, 2, 1)
        assert analyzer.compute_confidence_interval(single, k=1, seed=0) == (score, score)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1])
    def test_invalid_confidence_raises(self, pass_results_by_task, confidence):
        with pytest.raises(ValueError, match="confidence"):
            PassAtKAnalyzer().compute_confidence_interval(
                pass_results_by_task, k=1, confidence=confidence
            )

    def test_invalid_n_bootstrap_raises_even_for_empty_input(self):
        with pytest.raises(ValueError, match="n_bootstrap"):
            PassAtKAnalyzer().compute_confidence_interval({}, k=1, n_bootstrap=0)

    def test_analyze_with_ci_is_seeded_and_brackets_point(self, pass_results_by_task):
        analyzer = PassAtKAnalyzer(k_values=[1, 3, 5])
        first = analyzer.analyze_with_ci(pass_results_by_task, n_bootstrap=2000, seed=3)
        second = analyzer.analyze_with_ci(pass_results_by_task, n_bootstrap=2000, seed=3)
        assert first == second
        for entry in first.values():
            assert entry["lower"] <= entry["value"] <= entry["upper"]
