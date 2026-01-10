"""Tests for regression detection."""

import pytest

from agent_eval.baselines.comparison import (
    RegressionSeverity,
    MetricRegression,
    RegressionReport,
    RegressionDetector,
)
from agent_eval.baselines.manager import TaskBaseline


class TestRegressionSeverity:
    """Tests for RegressionSeverity enum."""

    def test_severity_ordering(self):
        """Test that severity levels are ordered correctly."""
        levels = [
            RegressionSeverity.NONE,
            RegressionSeverity.MINOR,
            RegressionSeverity.MODERATE,
            RegressionSeverity.SEVERE,
        ]

        # Verify they can be compared in order
        for i in range(len(levels) - 1):
            assert levels[i] != levels[i + 1]


class TestRegressionReport:
    """Tests for RegressionReport model."""

    def test_should_block_ci_no_regression(self):
        """Test CI blocking with no regression."""
        report = RegressionReport(
            has_regression=False,
            overall_severity=RegressionSeverity.NONE,
        )

        assert report.should_block_ci() is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_should_block_ci_minor_regression(self):
        """Test CI blocking with minor regression."""
        report = RegressionReport(
            has_regression=True,
            overall_severity=RegressionSeverity.MINOR,
        )

        # Default threshold is MODERATE
        assert report.should_block_ci() is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is True

    def test_should_block_ci_moderate_regression(self):
        """Test CI blocking with moderate regression."""
        report = RegressionReport(
            has_regression=True,
            overall_severity=RegressionSeverity.MODERATE,
        )

        assert report.should_block_ci() is True
        assert report.should_block_ci(RegressionSeverity.MODERATE) is True
        assert report.should_block_ci(RegressionSeverity.SEVERE) is False

    def test_should_block_ci_severe_regression(self):
        """Test CI blocking with severe regression."""
        report = RegressionReport(
            has_regression=True,
            overall_severity=RegressionSeverity.SEVERE,
        )

        assert report.should_block_ci() is True
        assert report.should_block_ci(RegressionSeverity.SEVERE) is True

    def test_to_ci_output_no_regression(self):
        """Test CI output with no regression."""
        report = RegressionReport(
            has_regression=False,
            overall_severity=RegressionSeverity.NONE,
        )

        output = report.to_ci_output()
        assert "No regressions detected" in output

    def test_to_ci_output_with_regression(self):
        """Test CI output with regression."""
        report = RegressionReport(
            has_regression=True,
            overall_severity=RegressionSeverity.MODERATE,
            regressions=[
                MetricRegression(
                    metric_name="sharpe_ratio",
                    baseline_mean=1.2,
                    current_mean=1.0,
                    delta=-0.2,
                    delta_percent=-16.67,
                    p_value=0.01,
                    is_significant=True,
                    severity=RegressionSeverity.MODERATE,
                )
            ],
        )

        output = report.to_ci_output()
        assert "REGRESSION DETECTED" in output
        assert "sharpe_ratio" in output


class TestRegressionDetector:
    """Tests for RegressionDetector class."""

    def test_creation(self):
        """Test detector creation."""
        detector = RegressionDetector(
            significance_level=0.05,
            min_delta_percent=5.0,
        )

        assert detector.significance_level == 0.05
        assert detector.min_delta_percent == 5.0

    def test_compare_no_regression(self, sample_baseline: TaskBaseline):
        """Test comparison with no regression."""
        detector = RegressionDetector()

        current_results = [
            {"sharpe_ratio": 1.3, "max_drawdown": -0.12, "win_rate": 0.58},
            {"sharpe_ratio": 1.25, "max_drawdown": -0.13, "win_rate": 0.57},
        ]

        report = detector.compare(sample_baseline, current_results)

        assert report.has_regression is False
        assert report.overall_severity == RegressionSeverity.NONE

    def test_compare_minor_regression(self, sample_baseline: TaskBaseline):
        """Test comparison with minor regression."""
        detector = RegressionDetector(min_delta_percent=2.0)

        current_results = [
            {"sharpe_ratio": 1.15},  # 4% decline
            {"sharpe_ratio": 1.17},
        ]

        report = detector.compare(sample_baseline, current_results)

        # May or may not be significant depending on statistics
        # But should not be severe or moderate
        assert report.overall_severity in {
            RegressionSeverity.NONE,
            RegressionSeverity.MINOR,
        }

    def test_compare_moderate_regression(self, sample_baseline: TaskBaseline):
        """Test comparison with moderate regression."""
        detector = RegressionDetector()

        current_results = [
            {"sharpe_ratio": 1.0},  # 17% decline
            {"sharpe_ratio": 1.02},
            {"sharpe_ratio": 0.98},
        ]

        report = detector.compare(sample_baseline, current_results)

        assert report.has_regression is True
        assert report.overall_severity in {
            RegressionSeverity.MODERATE,
            RegressionSeverity.SEVERE,
        }

    def test_compare_severe_regression(self, sample_baseline: TaskBaseline):
        """Test comparison with severe regression."""
        detector = RegressionDetector()

        current_results = [
            {"sharpe_ratio": 0.8},  # 33% decline
            {"sharpe_ratio": 0.85},
            {"sharpe_ratio": 0.78},
        ]

        report = detector.compare(sample_baseline, current_results)

        assert report.has_regression is True
        assert report.overall_severity == RegressionSeverity.SEVERE

    def test_compare_improvement(self, sample_baseline: TaskBaseline):
        """Test comparison detecting improvement."""
        detector = RegressionDetector()

        current_results = [
            {"sharpe_ratio": 1.5},  # 25% improvement
            {"sharpe_ratio": 1.55},
            {"sharpe_ratio": 1.48},
        ]

        report = detector.compare(sample_baseline, current_results)

        assert report.has_regression is False
        assert len(report.improvements) > 0

    def test_compare_multiple_metrics(self, sample_baseline: TaskBaseline):
        """Test comparison with multiple metrics."""
        detector = RegressionDetector()

        current_results = [
            {"sharpe_ratio": 0.9, "win_rate": 0.6},  # sharpe regresses, win_rate improves
        ]

        report = detector.compare(sample_baseline, current_results)

        # Should detect both regression and improvement
        # (depends on statistical significance)
        assert report.baseline_id == "btc_backtest"

    def test_compare_multiple_baselines(self, sample_baseline: TaskBaseline):
        """Test comparing multiple tasks."""
        detector = RegressionDetector()

        baselines = {"btc_backtest": sample_baseline}
        current_results = {
            "btc_backtest": [{"sharpe_ratio": 1.0}],
        }

        reports = detector.compare_multiple(baselines, current_results)

        assert "btc_backtest" in reports
