"""Tests for regression detection."""


from eval_kit.baselines.comparison import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
)
from eval_kit.baselines.manager import TaskBaseline


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


# --- Noise-aware regression detection (Track 2 / Anthropic infra-noise) ---


class TestNoiseAwareRegression:
    """Regression detection should distinguish real capability drops from
    deltas small enough to be explained by infrastructure-config drift,
    per Anthropic's "Quantifying infrastructure noise in agentic coding
    evals" (3pp default band)."""

    def _make_pass_rate_baseline(self, value: float = 0.75) -> TaskBaseline:
        baseline = TaskBaseline(task_id="some_task")
        baseline.add_metric(
            metric_name="pass_rate",
            value=value,
            std=0.02,
            sample_size=20,
            higher_is_better=True,
        )
        return baseline

    def test_noise_band_absolute_defaults_to_three_percentage_points(self):
        """The default threshold is 3pp, matching Anthropic's guidance."""
        detector = RegressionDetector()
        assert detector.noise_band_absolute == 0.03
        assert detector.noise_band_aware is True

    def test_compare_with_specs_no_specs_degrades_to_plain_compare(self):
        """When specs are not provided, compare_with_specs() must behave
        exactly like compare() — infra_config_mismatch stays False."""
        from eval_kit.baselines.comparison import RegressionDetector

        detector = RegressionDetector(min_delta_percent=1.0)
        baseline = self._make_pass_rate_baseline()
        report = detector.compare_with_specs(
            baseline,
            current_results=[{"pass_rate": 0.70}] * 5,
        )
        assert report.infra_config_mismatch is False
        assert report.infra_config_diff == {}

    def test_matching_infra_configs_do_not_trigger_noise_band(self):
        """Identical infra configs → infra_config_mismatch is False,
        and regressions retain their original severity even if small."""
        from eval_kit.baselines.comparison import RegressionDetector
        from eval_kit.core.decision_spec import DecisionSpec, InfraConfig

        detector = RegressionDetector(min_delta_percent=1.0)
        baseline = self._make_pass_rate_baseline()
        shared_infra = InfraConfig(cpu_hard_limit=3.0, memory_hard_limit_mb=2048)
        report = detector.compare_with_specs(
            baseline,
            current_results=[{"pass_rate": 0.72}] * 5,
            baseline_spec=DecisionSpec(infra=shared_infra),
            current_spec=DecisionSpec(infra=shared_infra),
        )
        assert report.infra_config_mismatch is False
        for reg in report.regressions:
            assert reg.within_noise_band is False

    def test_mismatched_infra_and_small_delta_flagged_within_noise_band(self):
        """Different infra + <3pp delta → regression is marked
        within_noise_band and drops from blocking_regressions."""
        from eval_kit.baselines.comparison import RegressionDetector
        from eval_kit.core.decision_spec import DecisionSpec, InfraConfig

        detector = RegressionDetector(min_delta_percent=1.0)
        # 2pp drop: 0.75 -> 0.73
        baseline = self._make_pass_rate_baseline(value=0.75)
        report = detector.compare_with_specs(
            baseline,
            current_results=[{"pass_rate": 0.73}] * 10,
            baseline_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=1.0)),
            current_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=3.0)),
        )
        assert report.infra_config_mismatch is True
        assert "cpu_hard_limit" in report.infra_config_diff
        assert report.infra_config_diff["cpu_hard_limit"] == (1.0, 3.0)
        # The 2pp regression is flagged as noise-band.
        assert len(report.regressions) == 1
        assert report.regressions[0].within_noise_band is True
        # blocking_regressions excludes it, so CI won't block by default.
        assert report.blocking_regressions == []
        assert report.should_block_ci() is False

    def test_mismatched_infra_but_large_delta_still_blocks(self):
        """A 10pp drop with mismatched infra should still register as a
        real regression — the noise-band applies only to small deltas."""
        from eval_kit.baselines.comparison import RegressionDetector
        from eval_kit.core.decision_spec import DecisionSpec, InfraConfig

        detector = RegressionDetector(min_delta_percent=1.0)
        baseline = self._make_pass_rate_baseline(value=0.75)
        # 10pp drop: 0.75 -> 0.65
        report = detector.compare_with_specs(
            baseline,
            current_results=[{"pass_rate": 0.65}] * 10,
            baseline_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=1.0)),
            current_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=3.0)),
        )
        assert report.infra_config_mismatch is True
        assert len(report.regressions) == 1
        assert report.regressions[0].within_noise_band is False
        assert len(report.blocking_regressions) == 1

    def test_should_block_ci_ignore_noise_band_false_still_blocks(self):
        """Callers can opt out of noise-band leniency by passing
        ignore_noise_band=False — then every regression counts even if
        the infra configs differ."""
        from eval_kit.baselines.comparison import RegressionDetector
        from eval_kit.core.decision_spec import DecisionSpec, InfraConfig

        detector = RegressionDetector(min_delta_percent=1.0)
        baseline = self._make_pass_rate_baseline(value=0.75)
        # 2pp drop → MINOR severity; sits inside the 3pp noise band.
        report = detector.compare_with_specs(
            baseline,
            current_results=[{"pass_rate": 0.73}] * 10,
            baseline_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=1.0)),
            current_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=3.0)),
        )
        # Lenient path at the MINOR threshold: the within-noise-band
        # regression is filtered out, so effective severity is NONE and
        # the check passes.
        assert report.should_block_ci(threshold=RegressionSeverity.MINOR) is False
        # Strict path at the same threshold: the regression still counts
        # as MINOR and the check blocks.
        assert report.should_block_ci(
            threshold=RegressionSeverity.MINOR,
            ignore_noise_band=False,
        ) is True

    def test_noise_band_aware_false_disables_flagging(self):
        """Setting noise_band_aware=False on the detector disables the
        downgrade entirely, even when specs mismatch."""
        from eval_kit.baselines.comparison import RegressionDetector
        from eval_kit.core.decision_spec import DecisionSpec, InfraConfig

        detector = RegressionDetector(
            min_delta_percent=1.0,
            noise_band_aware=False,
        )
        baseline = self._make_pass_rate_baseline(value=0.75)
        report = detector.compare_with_specs(
            baseline,
            current_results=[{"pass_rate": 0.73}] * 10,
            baseline_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=1.0)),
            current_spec=DecisionSpec(infra=InfraConfig(cpu_hard_limit=3.0)),
        )
        # No flag, no diff — we asked for raw behavior.
        assert report.infra_config_mismatch is False
        for reg in report.regressions:
            assert reg.within_noise_band is False
