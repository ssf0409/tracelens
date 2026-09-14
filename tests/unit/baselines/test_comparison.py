"""Tests for regression detection."""

import warnings

import pytest

from tracelens.baselines.comparison import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
    holm_adjusted,
)
from tracelens.baselines.manager import TaskBaseline


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

    def test_compare_degenerate_current_samples_without_runtime_warning(
        self,
        sample_baseline: TaskBaseline,
    ):
        """Zero-variance samples should not leak scipy precision warnings.

        A decisive drop (1.2 -> 0.2 against std=0.3) is decided by the
        pooled t-test on the baseline spread — with a real p-value, not
        the fabricated p=0.0 this path used to produce.
        """
        detector = RegressionDetector(min_delta_percent=1.0)
        current_results = [{"sharpe_ratio": 0.2}] * 5

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            report = detector.compare(sample_baseline, current_results)

        assert report.has_regression is True
        reg = next(r for r in report.regressions if r.metric_name == "sharpe_ratio")
        assert reg.p_value is not None
        assert reg.insufficient_data is False


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
        from tracelens.baselines.comparison import RegressionDetector

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
        from tracelens.baselines.comparison import RegressionDetector
        from tracelens.core.decision_spec import DecisionSpec, InfraConfig

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
        from tracelens.baselines.comparison import RegressionDetector
        from tracelens.core.decision_spec import DecisionSpec, InfraConfig

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
        from tracelens.baselines.comparison import RegressionDetector
        from tracelens.core.decision_spec import DecisionSpec, InfraConfig

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
        from tracelens.baselines.comparison import RegressionDetector
        from tracelens.core.decision_spec import DecisionSpec, InfraConfig

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
        from tracelens.baselines.comparison import RegressionDetector
        from tracelens.core.decision_spec import DecisionSpec, InfraConfig

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


class TestExactTestsOnPassRates:
    """Issue #111: 0/1 metrics get Boschloo's exact test on the two counts.

    The old fallback for a zero-variance baseline divided the delta by the
    sample SD (not the SE), so a 1.0 -> 0.4 drop over 5 trials had p=0.22
    at any n and was dropped from every output.
    """

    @staticmethod
    def _baseline(value: float = 1.0, std: float = 0.0, sample_size: int = 5) -> TaskBaseline:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="pass_rate", value=value, std=std, sample_size=sample_size)
        return baseline

    @staticmethod
    def _trials(passes: int, total: int) -> list[dict[str, float]]:
        return [{"pass_rate": 1.0 if i < passes else 0.0} for i in range(total)]

    def test_two_of_five_after_a_perfect_five_blocks(self) -> None:
        report = RegressionDetector().compare(self._baseline(), self._trials(2, 5))

        assert report.has_regression is True
        reg = report.regressions[0]
        assert reg.test == "boschloo_exact"
        assert reg.p_value == pytest.approx(0.0350, abs=5e-4)
        assert reg.is_significant and not reg.underpowered
        assert reg.severity is RegressionSeverity.SEVERE
        assert (reg.baseline_n, reg.current_n, reg.baseline_n_assumed) == (5, 5, False)
        assert report.should_block_ci(RegressionSeverity.MODERATE) is True
        assert "REGRESSION DETECTED [SEVERE]" in report.to_ci_output()
        assert "p=0.0350, significant" in report.to_ci_output()

    def test_three_of_five_is_reported_but_not_significant(self) -> None:
        report = RegressionDetector().compare(self._baseline(), self._trials(3, 5))

        # Reported (it is a 40% drop) but the evidence is not there, so it
        # neither counts as a detected regression nor blocks.
        assert len(report.regressions) == 1
        reg = report.regressions[0]
        assert reg.p_value == pytest.approx(0.1031, abs=5e-4)
        assert reg.is_significant is False and reg.underpowered is True
        assert reg.insufficient_data is False
        assert report.has_regression is False
        assert report.overall_severity is RegressionSeverity.NONE
        assert report.should_block_ci(RegressionSeverity.MINOR) is False
        text = report.to_ci_output()
        assert "No significant regression (1 observed drop(s)" in text
        assert "pass_rate: 1.0000 -> 0.6000 (-40.0%) [p=0.1031, not significant" in text
        # A 5-trial baseline caps the evidence any number of current trials
        # can give against it; the advice is to re-store it from more runs.
        assert reg.trials_needed_on_both_sides is True
        assert reg.trials_needed is not None and 5 <= reg.trials_needed <= 12
        assert "re-store it from more runs" in reg.evidence_text()

    def test_more_current_trials_tighten_the_test(self) -> None:
        detector = RegressionDetector()
        p_values = [
            detector.compare(self._baseline(), self._trials(passes, total)).regressions[0].p_value
            for passes, total in ((2, 5), (4, 10), (40, 100))
        ]
        assert all(p is not None for p in p_values)
        assert p_values[0] > p_values[1] > p_values[2]
        assert p_values[2] < 0.01

    def test_baseline_sample_size_drives_the_evidence(self) -> None:
        detector = RegressionDetector()
        small = detector.compare(self._baseline(sample_size=5), self._trials(3, 5))
        large = detector.compare(self._baseline(sample_size=10), self._trials(3, 5))

        assert small.regressions[0].is_significant is False
        assert large.regressions[0].is_significant is True
        assert large.regressions[0].p_value == pytest.approx(0.0380, abs=5e-4)

    def test_scaffold_style_std_does_not_change_a_count_test(self) -> None:
        detector = RegressionDetector()
        plain = detector.compare(self._baseline(std=0.0), self._trials(3, 5))
        scaffold = detector.compare(self._baseline(std=0.05), self._trials(3, 5))
        assert plain.regressions[0].p_value == scaffold.regressions[0].p_value

    def test_unrecorded_sample_size_is_assumed_equal_to_the_check(self) -> None:
        report = RegressionDetector().compare(self._baseline(sample_size=1), self._trials(2, 5))

        reg = report.regressions[0]
        assert reg.baseline_n_assumed is True and reg.baseline_n == 5
        assert reg.p_value == pytest.approx(0.0350, abs=5e-4)
        assert reg.is_significant

    def test_one_trial_against_a_declared_number_cannot_decide(self) -> None:
        report = RegressionDetector().compare(self._baseline(sample_size=1), self._trials(0, 1))

        reg = report.regressions[0]
        assert reg.p_value == pytest.approx(0.25)
        assert reg.is_significant is False and reg.underpowered is True
        assert reg.undetectable is True and reg.trials_needed == 3
        assert reg.severity is RegressionSeverity.SEVERE  # the size of the drop, as observed
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_total_failure_over_three_trials_blocks(self) -> None:
        report = RegressionDetector().compare(self._baseline(sample_size=1), self._trials(0, 3))
        reg = report.regressions[0]
        assert reg.p_value == pytest.approx(0.0156, abs=5e-4)
        assert reg.is_significant and report.should_block_ci()

    def test_p_values_are_one_sided_in_the_observed_direction(self) -> None:
        report = RegressionDetector().compare(self._baseline(0.6, sample_size=10), self._trials(5, 5))
        assert not report.regressions
        imp = report.improvements[0]
        assert imp.p_value == pytest.approx(0.0810, abs=5e-4)
        assert imp.is_significant is False and imp.trials_needed == 6
        assert "Improvements:" in report.to_ci_output()

    def test_lower_is_better_metrics_regress_upwards(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric("error_rate", 0.0, sample_size=10, higher_is_better=False)
        report = RegressionDetector().compare(
            baseline, [{"error_rate": 1.0}] * 4 + [{"error_rate": 0.0}]
        )
        assert report.has_regression is True
        assert report.regressions[0].delta > 0
        assert report.regressions[0].is_significant


class TestContinuousMetricTests:
    """Continuous metrics use the stored summary: Welch, pooled, or exact."""

    @staticmethod
    def _baseline(value: float, std: float, sample_size: int) -> TaskBaseline:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="mean_score", value=value, std=std, sample_size=sample_size)
        return baseline

    def test_welch_when_both_sides_have_a_spread(self) -> None:
        report = RegressionDetector().compare(
            self._baseline(0.9, 0.05, 20),
            [{"mean_score": v} for v in (0.5, 0.6, 0.55, 0.65, 0.5)],
        )
        reg = report.regressions[0]
        assert reg.test == "welch_t" and reg.is_significant
        assert reg.p_value is not None and reg.p_value < 1e-3
        assert reg.baseline_std == pytest.approx(0.05)
        assert reg.current_std == pytest.approx(0.0652, abs=1e-3)

    def test_pooled_when_the_current_sample_is_constant(self) -> None:
        # The old z-fallback case: z = 0.2 / (0.3 / sqrt(5)) ~ 1.49. Now the
        # drop is reported, marked not significant, and the note says how
        # many trials would decide it.
        report = RegressionDetector().compare(self._baseline(1.2, 0.3, 100), [{"mean_score": 1.0}] * 5)

        assert report.has_regression is False
        reg = report.regressions[0]
        assert reg.test == "pooled_t"
        assert reg.is_significant is False and reg.underpowered is True
        assert reg.trials_needed == 7 and reg.trials_needed_on_both_sides is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_exact_permutation_when_both_sides_are_constant(self) -> None:
        report = RegressionDetector().compare(self._baseline(1.0, 0.0, 10), [{"mean_score": 0.5}] * 5)
        reg = report.regressions[0]
        assert reg.test == "exact_permutation"
        assert reg.p_value == pytest.approx(1 / 3003)
        assert reg.is_significant and report.should_block_ci()

    def test_single_observation_against_a_baseline_sample(self) -> None:
        report = RegressionDetector().compare(self._baseline(1.0, 0.05, 10), [{"mean_score": 0.3}])
        reg = report.regressions[0]
        assert reg.test == "pooled_t" and reg.is_significant
        assert reg.current_n == 1 and reg.current_std is None

    def test_one_trial_against_a_declared_number_has_no_test(self) -> None:
        report = RegressionDetector().compare(self._baseline(0.9, 0.0, 1), [{"mean_score": 0.3}])
        reg = report.regressions[0]
        assert reg.p_value is None and reg.test is None
        assert reg.insufficient_data is True and reg.is_significant is False
        assert "no valid test (insufficient data)" in reg.evidence_text()
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_unrecorded_spread_is_taken_from_the_current_sample(self) -> None:
        report = RegressionDetector().compare(
            self._baseline(0.9, 0.0, 1),
            [{"mean_score": v} for v in (0.5, 0.6, 0.55, 0.65, 0.5)],
        )
        reg = report.regressions[0]
        assert reg.test == "welch_t" and reg.baseline_n_assumed
        assert reg.baseline_std is None and reg.is_significant


class TestBlockingPolicy:
    """Blocking needs severity at or above the threshold AND significance."""

    @staticmethod
    def _finding(**overrides: object) -> MetricRegression:
        values: dict[str, object] = {
            "metric_name": "pass_rate", "baseline_mean": 1.0, "current_mean": 0.6,
            "delta": -0.4, "delta_percent": -40.0, "p_value": 0.1, "is_significant": False,
            "severity": RegressionSeverity.SEVERE,
        }
        values.update(overrides)
        return MetricRegression(**values)  # type: ignore[arg-type]

    def test_underpowered_drop_never_blocks(self) -> None:
        report = RegressionReport(regressions=[self._finding()])
        report.recompute()
        assert report.has_regression is False
        assert report.overall_severity is RegressionSeverity.NONE
        assert report.should_block_ci(RegressionSeverity.MINOR) is False
        assert report.blocking_regressions == [] and report.underpowered_regressions

    def test_significant_drop_blocks_at_its_severity(self) -> None:
        report = RegressionReport(regressions=[self._finding(p_value=0.01, is_significant=True)])
        report.recompute()
        assert report.has_regression and report.overall_severity is RegressionSeverity.SEVERE
        assert report.should_block_ci(RegressionSeverity.SEVERE)

    def test_noise_band_still_exempts_significant_drops(self) -> None:
        finding = self._finding(p_value=0.01, is_significant=True, within_noise_band=True)
        report = RegressionReport(regressions=[finding])
        report.recompute()
        assert report.has_regression is True
        assert report.should_block_ci() is False
        assert report.should_block_ci(ignore_noise_band=False) is True

    def test_holm_adjusted_p_values(self) -> None:
        assert holm_adjusted([0.01, 0.04, 0.03, 1.0]) == pytest.approx([0.04, 0.09, 0.09, 1.0])
        assert holm_adjusted([0.2]) == [0.2]
        assert holm_adjusted([]) == []
        adjusted = holm_adjusted([0.001, 0.002, 0.5])
        assert adjusted == sorted(adjusted)  # monotone in the sorted order

    def test_evidence_text_states_adjusted_p_and_trials(self) -> None:
        finding = self._finding(p_value_adjusted=0.2, trials_needed=10)
        assert finding.evidence_text() == (
            "p=0.1000 (adjusted 0.2000), not significant; about 10 current trials "
            "would decide it"
        )
        assert self._finding(p_value=None).evidence_text() == "no valid test (insufficient data)"
        assert self._finding(trials_needed=None).evidence_text().endswith(
            "more than 200 trials would be needed"
        )


class TestSmallSampleHonesty:
    """Degenerate samples never fabricate significance and never block.

    With n=1 and no recorded spread or sample size (the model defaults)
    the evidence is what a single trial can give: a 0/1 metric gets the
    exact test at the assumed size and is undetectable; a continuous
    metric has no test at all. The drop is still reported.
    """

    def _degenerate_baseline(self, value: float = 1.0) -> TaskBaseline:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="pass_rate", value=value, std=0.0, sample_size=1)
        return baseline

    def test_single_trial_drop_is_reported_undetectable_and_not_blocking(self) -> None:
        report = RegressionDetector().compare(self._degenerate_baseline(), [{"pass_rate": 0.0}])

        assert report.has_regression is False
        reg = report.regressions[0]
        assert reg.severity == RegressionSeverity.SEVERE
        assert reg.p_value == pytest.approx(0.25)
        assert reg.is_significant is False and reg.undetectable is True
        assert report.should_block_ci(RegressionSeverity.MINOR) is False
        assert "not significant" in report.summary and "not significant" in report.to_ci_output()

    def test_two_identical_failures_are_still_undetectable(self) -> None:
        report = RegressionDetector().compare(
            self._degenerate_baseline(), [{"pass_rate": 0.0}, {"pass_rate": 0.0}]
        )
        reg = report.regressions[0]
        assert reg.p_value == pytest.approx(0.0625)
        assert reg.undetectable is True and reg.trials_needed == 3
        assert report.should_block_ci() is False

    def test_single_trial_against_a_measured_baseline_can_decide(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="pass_rate", value=1.0, std=0.05, sample_size=10)
        report = RegressionDetector().compare(baseline, [{"pass_rate": 0.0}])

        reg = report.regressions[0]
        assert reg.p_value == pytest.approx(0.0350, abs=5e-4)
        assert reg.is_significant is True and reg.insufficient_data is False

    def test_zero_variance_failures_against_a_measured_baseline(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="pass_rate", value=1.0, std=0.05, sample_size=20)
        report = RegressionDetector().compare(
            baseline, [{"pass_rate": 0.0}, {"pass_rate": 0.0}, {"pass_rate": 0.0}]
        )
        reg = report.regressions[0]
        assert reg.p_value is not None and reg.p_value < 0.001
        assert reg.insufficient_data is False and report.should_block_ci()


class TestReportedButNotSignificantPolicy:
    """A valid test that does not reject leaves the drop visible but not gating."""

    def test_consistent_but_nonsignificant_drop_is_reported_not_blocking(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(
            metric_name="mean_score", value=1.2, std=0.3, sample_size=100
        )
        detector = RegressionDetector()

        report = detector.compare(baseline, [{"mean_score": 1.0}] * 5)

        assert report.has_regression is False
        assert len(report.regressions) == 1
        assert report.regressions[0].underpowered is True
        assert report.should_block_ci(RegressionSeverity.MINOR) is False
        assert "not significant" in report.summary


class TestNoiseAwareSeverityConsistency:
    """After the noise-band downgrade, overall_severity must agree with
    the blocking decision — a noise-only report must not keep shouting
    SEVERE while should_block_ci() returns False."""

    def _specs(self) -> tuple:
        from tracelens.core.decision_spec import DecisionSpec, InfraConfig

        return (
            DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=2048)),
            DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=512)),
        )

    def test_noise_only_report_downgrades_overall_severity(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(
            metric_name="pass_rate", value=0.10, std=0.001, sample_size=20
        )
        baseline_spec, current_spec = self._specs()
        detector = RegressionDetector()

        # -20% relative (SEVERE by thresholds) but only 0.02 absolute:
        # inside the 0.03 noise band under mismatched infra.
        report = detector.compare_with_specs(
            baseline,
            [{"pass_rate": 0.08}],
            baseline_spec=baseline_spec,
            current_spec=current_spec,
        )

        assert report.regressions[0].within_noise_band is True
        assert report.should_block_ci() is False
        assert report.overall_severity == RegressionSeverity.NONE
        assert "noise band" in report.summary.lower()
        # Still surfaced — has_regression reports what was observed.
        assert report.has_regression is True
