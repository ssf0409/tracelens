"""Tests for regression detection."""

import warnings

import pytest

from tracelens.baselines.comparison import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
    holm_adjusted,
    relative_change,
    severity_at_least,
)
from tracelens.baselines.manager import MetricBaseline, TaskBaseline


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

    def test_severity_at_least_and_relative_change(self):
        assert severity_at_least(RegressionSeverity.SEVERE, RegressionSeverity.MODERATE)
        assert severity_at_least(RegressionSeverity.MODERATE, RegressionSeverity.MODERATE)
        assert not severity_at_least(RegressionSeverity.MINOR, RegressionSeverity.MODERATE)
        assert relative_change(-0.2, 0.8) == pytest.approx(-25.0)
        assert relative_change(0.1, -0.5) == pytest.approx(20.0)
        assert relative_change(0.3, 0.0) == 100.0 and relative_change(0.0, 0.0) == 0.0


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

        A decisive drop (1.2 -> 0.2 against std=0.3) is decided by Welch
        on the baseline's spread, which stands for the flat current sample
        too — with a real p-value, not the fabricated p=0.0 this path used
        to produce.
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
        # scipy.stats.boschloo_exact([[5, 2], [0, 3]], alternative="greater")
        # -- the two samples are the columns, as SciPy documents.
        assert reg.p_value == pytest.approx(0.0309, abs=5e-4)
        assert reg.is_significant and not reg.underpowered
        assert reg.severity is RegressionSeverity.SEVERE
        assert (reg.baseline_n, reg.current_n, reg.baseline_n_assumed) == (5, 5, False)
        assert report.should_block_ci(RegressionSeverity.MODERATE) is True
        assert "REGRESSION DETECTED [SEVERE]" in report.to_ci_output()
        assert "p=0.0309, significant" in report.to_ci_output()

    def test_three_of_five_is_reported_but_not_significant(self) -> None:
        report = RegressionDetector().compare(self._baseline(), self._trials(3, 5))

        # Reported (it is a 40% drop) but the evidence is not there, so it
        # neither counts as a detected regression nor blocks.
        assert len(report.regressions) == 1
        reg = report.regressions[0]
        # boschloo_exact([[5, 3], [0, 2]], alternative="greater")
        assert reg.p_value == pytest.approx(0.0937, abs=5e-4)
        assert reg.is_significant is False and reg.underpowered is True
        assert reg.insufficient_data is False
        assert report.has_regression is False
        assert report.overall_severity is RegressionSeverity.NONE
        assert report.should_block_ci(RegressionSeverity.MINOR) is False
        text = report.to_ci_output()
        assert "No significant regression (1 observed drop(s)" in text
        assert "pass_rate: 1.0000 -> 0.6000 (-40.0%) [p=0.0937, not significant" in text
        # The advice says what would decide the rates actually observed:
        # 19 current trials against the stored 5.
        assert reg.trials_needed == 19 and reg.trials_needed_on_both_sides is False
        assert "about 19 current trials would decide it" in reg.evidence_text()

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
        # boschloo_exact([[10, 3], [0, 2]], alternative="greater")
        assert large.regressions[0].p_value == pytest.approx(0.0460, abs=5e-4)

    def test_scaffold_style_std_does_not_change_a_count_test(self) -> None:
        detector = RegressionDetector()
        plain = detector.compare(self._baseline(std=0.0), self._trials(3, 5))
        scaffold = detector.compare(self._baseline(std=0.05), self._trials(3, 5))
        assert plain.regressions[0].p_value == scaffold.regressions[0].p_value

    def test_a_declared_baseline_is_never_credited_with_runs_it_never_had(self) -> None:
        # A baseline that stored one trial is a declared value, not five
        # runs of evidence. Inflating its size to match the check turned
        # 1-of-1 into 5-of-5 and blocked on a comparison that never
        # happened; the honest table is 1/1 against 2/5.
        report = RegressionDetector().compare(self._baseline(sample_size=1), self._trials(2, 5))

        reg = report.regressions[0]
        assert reg.baseline_n_assumed is True and reg.baseline_n == 1
        assert reg.baseline_std is None  # one trial measures no spread
        # boschloo_exact([[1, 2], [0, 3]], alternative="greater")
        assert reg.p_value == pytest.approx(0.2730, abs=5e-4)
        assert reg.is_significant is False

    def test_one_trial_against_a_declared_number_cannot_decide(self) -> None:
        report = RegressionDetector().compare(self._baseline(sample_size=1), self._trials(0, 1))

        reg = report.regressions[0]
        assert reg.p_value == pytest.approx(0.25)
        assert reg.is_significant is False and reg.underpowered is True
        assert reg.undetectable is True and reg.trials_needed == 7
        assert reg.severity is RegressionSeverity.SEVERE  # the size of the drop, as observed
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_a_declared_baseline_cannot_decide_a_total_failure_but_a_measured_one_can(
        self,
    ) -> None:
        # boschloo_exact([[1, 0], [0, 3]], "greater") = 0.1055: one stored
        # trial cannot carry a run-blocking verdict, however total the
        # failure. Five stored trials can: [[5, 0], [0, 3]] = 0.0050.
        declared = RegressionDetector().compare(
            self._baseline(sample_size=1), self._trials(0, 3)
        ).regressions[0]
        assert declared.p_value == pytest.approx(0.1055, abs=5e-4)
        assert declared.is_significant is False

        measured = RegressionDetector().compare(
            self._baseline(sample_size=5), self._trials(0, 3)
        )
        assert measured.regressions[0].p_value == pytest.approx(0.0050, abs=5e-4)
        assert measured.regressions[0].is_significant and measured.should_block_ci()

    def test_p_values_are_one_sided_in_the_observed_direction(self) -> None:
        report = RegressionDetector().compare(self._baseline(0.6, sample_size=10), self._trials(5, 5))
        assert not report.regressions
        imp = report.improvements[0]
        # boschloo_exact([[6, 5], [4, 0]], alternative="less")
        assert imp.p_value == pytest.approx(0.0647, abs=5e-4)
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


class TestTheTableSciPyExpects:
    """Regression tests for the transposed Boschloo table.

    SciPy's model puts one binomial experiment in each column. The earlier
    implementation passed the two samples as rows, which fixes the wrong
    margin and answers a different question; at unequal sample sizes the
    two orientations disagree enough to flip a verdict.
    """

    @staticmethod
    def _baseline(value: float, sample_size: int) -> TaskBaseline:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(
            metric_name="pass_rate", value=value, std=0.0, sample_size=sample_size
        )
        return baseline

    @staticmethod
    def _trials(passes: int, total: int) -> list[dict[str, float]]:
        return [{"pass_rate": 1.0 if i < passes else 0.0} for i in range(total)]

    def test_unequal_sample_sizes_use_the_orientation_scipy_documents(self) -> None:
        # Seven stored trials, all passing, against two of four now.
        # boschloo_exact([[7, 2], [0, 2]], alternative="greater") = 0.05378,
        # which does not reach alpha. Transposed it reads 0.04205 and blocks.
        report = RegressionDetector().compare(self._baseline(1.0, 7), self._trials(2, 4))

        reg = report.regressions[0]
        assert reg.test == "boschloo_exact"
        assert reg.p_value == pytest.approx(0.0538, abs=5e-4)
        assert reg.is_significant is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_a_rise_is_the_mirror_of_the_matching_drop(self) -> None:
        # The same table read the other way round: 2/5 -> 5/5 as an
        # improvement carries the p-value of 5/5 -> 2/5 as a drop.
        drop = RegressionDetector().compare(
            self._baseline(1.0, 5), self._trials(2, 5)
        ).regressions[0]
        rise = RegressionDetector().compare(
            self._baseline(0.4, 5), self._trials(5, 5)
        ).improvements[0]
        assert drop.p_value == pytest.approx(rise.p_value)
        assert drop.p_value == pytest.approx(0.0309, abs=5e-4)

    def test_a_declared_baseline_cannot_manufacture_a_block(self) -> None:
        # The reported case: one stored trial that passed, against eighty of
        # a hundred now. Inflating the baseline to the check size evaluated
        # it as 100/100 against 80/100 and blocked at p=1.3e-07. The honest
        # table is boschloo_exact([[1, 80], [0, 20]], "greater") = 0.70137.
        report = RegressionDetector().compare(
            self._baseline(1.0, 1),
            [{"pass_rate": 1.0}] * 80 + [{"pass_rate": 0.0}] * 20,
        )

        reg = report.regressions[0]
        assert reg.baseline_n == 1 and reg.current_n == 100
        assert reg.p_value == pytest.approx(0.7014, abs=5e-4)
        assert reg.is_significant is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is False


class TestADeclarationCannotSupplyMissingEvidence:
    """`is_rate` says which family a metric belongs to, nothing more."""

    @staticmethod
    def _declared(value: float, sample_size: int) -> TaskBaseline:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(
            metric_name="pass_rate", value=value, std=0.0,
            sample_size=sample_size, is_rate=True,
        )
        return baseline

    def test_a_declared_rate_still_needs_a_whole_count(self) -> None:
        # 0.9 over five trials is four and a half successes. Rounding it to
        # a count makes the baseline perfect (5 of 5) and blocks at 0.0309;
        # truncating makes it 4 of 5 and reads 0.1719. Neither is measured,
        # so the exact test does not apply and the comparison falls back to
        # the continuous path.
        report = RegressionDetector().compare(
            self._declared(0.9, 5), [{"pass_rate": v} for v in (1.0, 1.0, 0.0, 0.0, 0.0)]
        )
        assert report.regressions[0].test != "boschloo_exact"

    def test_a_declared_rate_with_a_real_count_uses_the_exact_test(self) -> None:
        report = RegressionDetector().compare(
            self._declared(0.8, 5), [{"pass_rate": v} for v in (1.0, 1.0, 0.0, 0.0, 0.0)]
        )
        assert report.regressions[0].test == "boschloo_exact"

    def test_a_declared_rate_with_a_nan_mean_is_no_evidence(self) -> None:
        # A corrupt value must not reach a test: two "constant" sides, one
        # of them NaN, land on the permutation value and read as the
        # strongest evidence the sample sizes allow.
        report = RegressionDetector().compare(
            self._declared(float("nan"), 5), [{"pass_rate": 0.0}] * 5
        )
        finding = (report.regressions or report.improvements)[0]
        assert finding.test is None and finding.p_value is None
        assert finding.is_significant is False and finding.undetectable is True
        assert report.should_block_ci(RegressionSeverity.MINOR) is False


class TestMetricTypeComesFromTheBaseline:
    """The test family must not depend on where the current sample lands."""

    @staticmethod
    def _continuous() -> TaskBaseline:
        # A tenth of a success over five trials is not a count, so this
        # summary cannot be a proportion however the current values look.
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="score", value=0.1, std=0.01, sample_size=5)
        return baseline

    def test_an_infinitesimal_change_cannot_switch_the_test_family(self) -> None:
        detector = RegressionDetector()
        zeros = detector.compare(self._continuous(), [{"score": 0.0}] * 5).regressions[0]
        almost = detector.compare(self._continuous(), [{"score": 1e-8}] * 5).regressions[0]

        # Same family, and p-values that differ only by the 1e-8 itself.
        assert zeros.test == almost.test == "welch_t"
        assert zeros.p_value == pytest.approx(almost.p_value, rel=1e-3)
        assert zeros.is_significant == almost.is_significant

    def test_a_scaffolded_pass_rate_is_still_a_proportion(self) -> None:
        # ``tracelens init`` records a nominal spread it never measured.
        # That is not evidence against a rate, so the exact test still runs.
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric("pass_rate", 1.0, std=0.05, sample_size=5)
        reg = RegressionDetector().compare(
            baseline, [{"pass_rate": v} for v in (1.0, 1.0, 0.0, 0.0, 0.0)]
        ).regressions[0]
        assert reg.test == "boschloo_exact"

    def test_a_corrupt_sample_size_is_no_evidence_and_does_not_crash(self) -> None:
        # A negative count handed to the exact test raises out of SciPy and
        # takes the whole run down. A corrupt record is not evidence: the
        # comparison has no valid test, so the check is undetectable and
        # nothing about it can block.
        for size in (0, -1):
            baseline = TaskBaseline(task_id="t1")
            baseline.metrics["pass_rate"] = MetricBaseline(
                metric_name="pass_rate", baseline_value=1.0, sample_size=size
            )
            report = RegressionDetector().compare(baseline, [{"pass_rate": 0.0}] * 3)
            reg = report.regressions[0]
            assert reg.test is None and reg.p_value is None
            assert reg.insufficient_data is True and reg.undetectable is True
            assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_a_baseline_may_declare_what_it_is(self) -> None:
        values = [{"pass_rate": v} for v in (1.0, 1.0, 0.0, 0.0, 0.0)]
        declared = TaskBaseline(task_id="t1")
        declared.add_metric("pass_rate", 1.0, std=0.05, sample_size=5, is_rate=False)
        reg = RegressionDetector().compare(declared, values).regressions[0]
        assert reg.test == "welch_t"  # the declaration wins over the inference


class TestContinuousMetricTests:
    """Continuous metrics use the stored summary: Welch or exact.

    The pooled-variance test is not used at all: a spread measured as zero
    on one side does not show that the two populations share a variance.
    """

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

    def test_a_flat_sample_does_not_get_credited_with_zero_uncertainty(self) -> None:
        # A hundred baseline trials with sd 0.3 against five that all read
        # 1.0. Five trials that happen to agree are not a measurement of
        # zero variance, and handing that zero to Welch would credit the
        # current mean with no uncertainty at all. The spread that WAS
        # informative stands for both sides:
        # ttest_ind_from_stats(1.2, 0.3, 100, 1.0, 0.3, 5, equal_var=False,
        # alternative="greater") = 0.10650.
        report = RegressionDetector().compare(self._baseline(1.2, 0.3, 100), [{"mean_score": 1.0}] * 5)

        reg = report.regressions[0]
        assert reg.test == "welch_t"
        # The finding still records what each side actually measured.
        assert reg.baseline_std == pytest.approx(0.3) and reg.current_std == pytest.approx(0.0)
        assert reg.p_value == pytest.approx(0.1065, abs=5e-4)
        assert reg.is_significant is False and report.has_regression is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_a_zero_baseline_spread_no_longer_pools(self) -> None:
        # The reported case: baseline mean .5 recorded with sd 0 over 100
        # trials, checked against three scattered values. Pooling gave
        # p=6e-34 and blocked, because a flat side with the larger n
        # dominates the pooled variance estimate. Using the informative
        # spread for both sides instead:
        # ttest_ind_from_stats(.5, 0.2, 100, .2, 0.2, 3, equal_var=False,
        # alternative="greater") = 0.05880, which does not block.
        report = RegressionDetector().compare(
            self._baseline(0.5, 0.0, 100),
            [{"mean_score": v} for v in (0.0, 0.2, 0.4)],
        )
        reg = report.regressions[0]
        assert reg.test == "welch_t"
        assert reg.p_value == pytest.approx(0.0588, abs=5e-4)
        assert reg.is_significant is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_two_constant_sides_at_the_same_value_are_no_evidence(self) -> None:
        # The permutation value is the probability of the MOST extreme
        # split, so it only applies when the two sides actually separate.
        # Two constant sides at the same value do not, and reading
        # 1/C(8, 4) there would make "nothing changed" the strongest
        # evidence the sample sizes allow.
        report = RegressionDetector(min_delta_percent=0.0).compare(
            self._baseline(0.9, 0.0, 4), [{"mean_score": 0.9}] * 4
        )
        finding = (report.regressions or report.improvements)[0]
        assert finding.delta == 0.0
        assert finding.p_value == 1.0 and finding.is_significant is False
        assert report.has_regression is False

    def test_exact_permutation_when_both_sides_are_constant(self) -> None:
        report = RegressionDetector().compare(self._baseline(1.0, 0.0, 10), [{"mean_score": 0.5}] * 5)
        reg = report.regressions[0]
        assert reg.test == "exact_permutation"
        assert reg.p_value == pytest.approx(1 / 3003)
        assert reg.is_significant and report.should_block_ci()

    def test_single_observation_against_a_baseline_sample(self) -> None:
        # One new observation against a measured baseline is a prediction
        # interval, not a two-sample t-test with a fabricated spread of zero.
        report = RegressionDetector().compare(self._baseline(1.0, 0.05, 10), [{"mean_score": 0.3}])
        reg = report.regressions[0]
        assert reg.test == "prediction_t" and reg.is_significant
        assert reg.current_n == 1 and reg.current_std is None

    def test_one_trial_against_a_declared_number_has_no_test(self) -> None:
        report = RegressionDetector().compare(self._baseline(0.9, 0.0, 1), [{"mean_score": 0.3}])
        reg = report.regressions[0]
        assert reg.p_value is None and reg.test is None
        assert reg.insufficient_data is True and reg.is_significant is False
        assert "no valid test (insufficient data)" in reg.evidence_text()
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_an_unrecorded_spread_is_not_borrowed_from_the_current_sample(self) -> None:
        # A baseline of one trial measured no spread. Standing the current
        # sample's spread in for it manufactures the evidence the test then
        # reports, so there is no valid test at all.
        report = RegressionDetector().compare(
            self._baseline(0.9, 0.0, 1),
            [{"mean_score": v} for v in (0.5, 0.6, 0.55, 0.65, 0.5)],
        )
        reg = report.regressions[0]
        assert reg.test is None and reg.p_value is None
        assert reg.baseline_n_assumed and reg.baseline_std is None
        assert reg.insufficient_data is True and reg.is_significant is False
        assert reg.undetectable is True


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
        # "More than the cap" is a result of the power scan, so it is stated
        # only when the scan actually ran and came back empty-handed.
        assert self._finding(
            trials_needed=None, trials_needed_exceeds_cap=True
        ).evidence_text().endswith("more than 200 trials would be needed")
        # A finding that carries no power analysis at all -- every artifact
        # written before these fields existed, and every run with
        # power_notes=False -- must not assert one it never performed.
        assert self._finding(trials_needed=None).evidence_text() == (
            "p=0.1000, not significant"
        )

    def test_power_notes_can_be_switched_off(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="pass_rate", value=1.0, std=0.0, sample_size=5)
        trials = [{"pass_rate": 1.0}] * 3 + [{"pass_rate": 0.0}] * 2

        quiet = RegressionDetector(power_notes=False).compare(baseline, trials).regressions[0]
        assert quiet.p_value == pytest.approx(0.0937, abs=5e-4)
        assert quiet.is_significant is False and quiet.underpowered is True
        assert quiet.trials_needed is None and quiet.undetectable is False

        # The notes are computed on request, at the level the caller names.
        RegressionDetector().annotate(quiet, 0.05)
        assert quiet.trials_needed == 19
        full = RegressionDetector().compare(baseline, trials).regressions[0]
        assert full.trials_needed == 19 and full.evidence_text() == quiet.evidence_text()


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
        # boschloo_exact([[1, 0], [0, 2]], alternative="greater")
        assert reg.p_value == pytest.approx(0.1481, abs=5e-4)
        assert reg.undetectable is True and reg.trials_needed == 7
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

    def test_a_declared_rate_that_is_not_a_count_is_no_evidence(self) -> None:
        # A declared 10% rate over one stored trial: a tenth of a success is
        # not a count, so this summary is not a proportion and the exact
        # test does not apply; one trial also measured no spread, so no
        # t-test applies either. The drop is reported with no evidence
        # behind it, and nothing about it can block.
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="pass_rate", value=0.1, std=0.0, sample_size=1)
        report = RegressionDetector().compare(baseline, [{"pass_rate": 0.0}] * 4)

        reg = report.regressions[0]
        assert reg.test is None and reg.p_value is None
        assert reg.insufficient_data is True
        assert reg.is_significant is False and reg.undetectable is True
        assert report.has_regression is False
        assert report.should_block_ci(RegressionSeverity.MINOR) is False

    def test_lower_is_better_power_notes_follow_the_metric_direction(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric("error_rate", 0.0, sample_size=1, higher_is_better=False)
        detector = RegressionDetector()

        report = detector.compare(baseline, [{"error_rate": 1.0}])
        reg = report.regressions[0]
        assert reg.higher_is_better is False and reg.p_value == pytest.approx(0.25)
        assert reg.is_significant is False
        # The worst case for this metric is every trial at 1.0, which is what
        # was observed: still not significant, so the drop is undetectable.
        assert reg.undetectable is True and reg.trials_needed == 7
        metric = baseline.get_metric("error_rate")
        assert metric is not None
        assert detector.detectability(metric, [1.0], 0.05) == (False, 7)

        # Three trials against one stored trial still cannot decide it
        # (boschloo_exact([[0, 3], [1, 0]], alternative="less") = 0.1055);
        # a baseline stored from five clean runs can.
        declared = detector.compare(baseline, [{"error_rate": 1.0}] * 3).regressions[0]
        assert declared.p_value == pytest.approx(0.1055, abs=5e-4)
        assert declared.is_significant is False

        measured = TaskBaseline(task_id="t1")
        measured.add_metric("error_rate", 0.0, sample_size=5, higher_is_better=False)
        decided = detector.compare(measured, [{"error_rate": 1.0}] * 3).regressions[0]
        assert decided.p_value == pytest.approx(0.0050, abs=5e-4)
        assert decided.is_significant is True and decided.undetectable is False


class TestTheBaselineIsWhatLimitsTheEvidence:
    """A baseline of one stored trial must say so, not "more than 200"."""

    def test_a_declared_baseline_gets_the_re_store_advice(self) -> None:
        # Growing the check alone cannot decide this at the level the gate
        # uses, so the note has to name the baseline as the limit. The
        # branch that says so used to be skipped for exactly these findings.
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(metric_name="pass_rate", value=1.0, std=0.0, sample_size=1)
        detector = RegressionDetector()
        finding = detector.compare(baseline, [{"pass_rate": 0.0}] * 2).regressions[0]
        detector.annotate(finding, 0.05, 0.001)

        assert finding.trials_needed is not None
        assert finding.trials_needed_on_both_sides is True
        assert "trials on each side would decide it" in finding.evidence_text()
        assert "re-store it from more runs" in finding.evidence_text()


class TestReportedButNotSignificantPolicy:
    """A valid test that does not reject leaves the drop visible but not gating."""

    def test_consistent_but_nonsignificant_drop_is_reported_not_blocking(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        baseline.add_metric(
            metric_name="mean_score", value=1.2, std=0.3, sample_size=5
        )
        detector = RegressionDetector()

        report = detector.compare(
            baseline, [{"mean_score": v} for v in (1.0, 1.2, 1.1, 1.05, 1.15)]
        )

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


class TestAZeroSpreadIsRecognisedWhateverTheValue:
    """``np.std`` of a constant sample is exactly 0.0 only for some values.

    Five 0.5s give 0.0; three 0.7s give 1.36e-16. An exact ``== 0.0`` test
    therefore fired for some stored numbers and not others, so the
    substitution that keeps a flat side from being credited with zero
    uncertainty silently did not happen and Welch ran with a spread of
    ~1e-16. The verdict then turned on the sixteenth decimal of an
    arbitrary value, which reads as flakiness in the field.
    """

    @staticmethod
    def _sides(mean_c: float, std_c: float, n_c: int = 3):
        from tracelens.baselines.comparison import _Sides

        return _Sides(
            binary=False, mean_b=0.70, n_b=5, std_b=0.10,
            mean_c=mean_c, n_c=n_c, std_c=std_c,
            baseline_n_assumed=False, higher_is_better=True,
        )

    def test_a_constant_sample_is_flat_whatever_value_it_repeats(self) -> None:
        import numpy as np

        from tracelens.baselines.comparison import _is_flat

        # Not a hypothetical: these are what numpy actually returns.
        assert float(np.std([0.5] * 3, ddof=1)) == 0.0
        assert float(np.std([0.7] * 3, ddof=1)) > 0.0
        for value in (0.5, 0.25, 0.7, 0.2, 0.1, 0.3, 0.6):
            for n in (3, 5, 20):
                assert _is_flat(float(np.std([value] * n, ddof=1)))

    def test_the_verdict_does_not_turn_on_the_sixteenth_decimal(self) -> None:
        import numpy as np

        from tracelens.baselines.comparison import _p_value

        dust = float(np.std([0.7] * 3, ddof=1))
        assert dust > 0.0  # the value that used to slip past the guard
        for mean_c in (0.65, 0.60, 0.55, 0.50):
            _, p_clean = _p_value(self._sides(mean_c, 0.0), "greater")
            _, p_dusty = _p_value(self._sides(mean_c, dust), "greater")
            assert p_clean == pytest.approx(p_dusty, rel=1e-6)

    def test_a_corrupt_spread_takes_the_no_variation_path(self) -> None:
        from tracelens.baselines.comparison import _is_flat

        # Neither a negative nor a NaN spread is a measurement; both must
        # reach the no-variation branch rather than a test that divides by
        # them, which returned p=0.0 for a constant check sample.
        assert _is_flat(-5.0) and _is_flat(float("nan"))


class TestRerunAdviceIsNotFlooredByTheCurrentSize:
    """When the baseline is re-stored too, a smaller size can decide it.

    The scan's floor at the current size is justified only while the
    baseline is held fixed. With both sides growing it put a floor of
    ``n_c + 1`` on the answer, so a one-trial baseline against 80 of 100
    was told to rerun about 101 trials per side where 18 settles it.
    """

    def test_both_sides_advice_matches_an_independent_scan(self) -> None:
        from scipy import stats

        from tracelens.baselines.comparison import _Sides, _trials_needed

        sides = _Sides(
            binary=True, mean_b=1.0, n_b=1, std_b=None,
            mean_c=0.8, n_c=100, std_c=0.4020151,
            baseline_n_assumed=True, higher_is_better=True,
        )
        # Derived from scipy, not from the implementation: the smallest n
        # where an n-of-n baseline against round(0.8n)-of-n is significant.
        expected = next(
            n for n in range(2, 60)
            if float(stats.boschloo_exact(
                [[n, int(0.8 * n + 0.5)], [0, n - int(0.8 * n + 0.5)]],
                alternative="greater",
            ).pvalue) <= 0.05
        )
        assert expected == 18
        assert _trials_needed(sides, 0.05, both_sides=True) == expected
