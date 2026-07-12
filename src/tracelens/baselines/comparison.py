"""Regression detection and comparison.

Compares current evaluation results against baselines to detect
performance regressions.
"""

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from tracelens.baselines.manager import TaskBaseline
from tracelens.core._time import utc_now
from tracelens.core.decision_spec import DecisionSpec

# Default noise band in absolute metric units — i.e. 0.03 means "a score
# change of 3 percentage points on a 0-1 metric." Comes from Anthropic's
# "Quantifying infrastructure noise in agentic coding evals" (Feb 2026):
# "Until resource methodology is standardized, our data suggests that
# leaderboard differences below 3 percentage points deserve skepticism
# until the eval configuration is documented and matched."
DEFAULT_NOISE_BAND_ABSOLUTE: float = 0.03


class RegressionSeverity(str, Enum):
    """Severity levels for regressions."""

    NONE = "none"           # No regression
    MINOR = "minor"         # < 5% decline
    MODERATE = "moderate"   # 5-15% decline (default blocking threshold)
    SEVERE = "severe"       # > 15% decline


class MetricRegression(BaseModel):
    """Detected regression in a specific metric."""

    metric_name: str
    baseline_mean: float
    current_mean: float
    delta: float
    delta_percent: float

    # Statistical significance. p_value is None when no valid test exists
    # (zero variance on both sides of the comparison) — never a fabricated
    # 0.0. Such regressions carry insufficient_data=True and their severity
    # rests on the delta thresholds alone.
    p_value: float | None
    is_significant: bool
    insufficient_data: bool = False

    severity: RegressionSeverity

    # Which tasks were affected
    affected_tasks: list[str] = Field(default_factory=list)

    # Noise-awareness: True if the absolute delta falls within the noise
    # band AND the baseline/current infra configurations don't match.
    # Deltas flagged this way are surfaced but do NOT block CI by default
    # (see RegressionReport.blocking_regressions).
    within_noise_band: bool = False


class RegressionReport(BaseModel):
    """Complete regression analysis report."""

    baseline_id: str | None = None
    baseline_commit: str | None = None
    current_commit: str | None = None

    # Overall assessment
    has_regression: bool = False
    overall_severity: RegressionSeverity = RegressionSeverity.NONE

    # Detailed regressions
    regressions: list[MetricRegression] = Field(default_factory=list)

    # Improvements (optional tracking)
    improvements: list[MetricRegression] = Field(default_factory=list)

    # Summary
    summary: str = ""

    # Metadata
    generated_at: datetime = Field(default_factory=utc_now)

    # --- Noise awareness -------------------------------------------------
    # True when the baseline's DecisionSpec.infra differs from the current
    # run's. Deltas below the noise band in this state are treated as
    # "could be infra noise, not a real regression" (Anthropic, Feb 2026).
    infra_config_mismatch: bool = False

    # Raw diff of the two infra configs (baseline_value, current_value).
    # Empty dict when configs match or when no specs were provided.
    infra_config_diff: dict[str, tuple[Any, Any]] = Field(default_factory=dict)

    @property
    def blocking_regressions(self) -> list[MetricRegression]:
        """Regressions that should actually block CI.

        Excludes any regression marked ``within_noise_band`` — those are
        within the infra-noise uncertainty and shouldn't gate merges
        until the eval configuration is matched.
        """
        return [r for r in self.regressions if not r.within_noise_band]

    def should_block_ci(
        self,
        threshold: RegressionSeverity = RegressionSeverity.MODERATE,
        ignore_noise_band: bool = True,
    ) -> bool:
        """Determine if CI should be blocked based on severity.

        Args:
            threshold: Minimum severity to block. Default: MODERATE
            ignore_noise_band: If True (default), within-noise-band
                regressions don't count — a 2pp drop under a mismatched
                infra config is ambiguous and shouldn't auto-gate merges
                per Anthropic's infra-noise guidance. Pass False to treat
                every regression as blocking regardless of noise.

        Returns:
            True if CI should be blocked
        """
        severity_order = [
            RegressionSeverity.NONE,
            RegressionSeverity.MINOR,
            RegressionSeverity.MODERATE,
            RegressionSeverity.SEVERE,
        ]
        # When the regressions list is populated, recompute severity from
        # the filtered list so noise-band-flagged entries drop out. When
        # it's empty (e.g. a hand-constructed report where the caller
        # only set overall_severity), fall back to the declared severity
        # so existing callers keep working.
        if ignore_noise_band and self.regressions:
            effective_severity = max(
                (r.severity for r in self.blocking_regressions),
                default=RegressionSeverity.NONE,
            )
        else:
            effective_severity = self.overall_severity
        return severity_order.index(effective_severity) >= severity_order.index(threshold)

    def to_ci_output(self) -> str:
        """Generate CI-friendly output."""
        lines = []

        if self.has_regression:
            lines.append(f"REGRESSION DETECTED [{self.overall_severity.value.upper()}]")
            lines.append("")

            for reg in self.regressions:
                notes = ""
                if reg.insufficient_data:
                    notes += " [insufficient samples for significance; severity from thresholds]"
                if reg.within_noise_band:
                    notes += " [within infra-noise band; not blocking]"
                lines.append(
                    f"  {reg.metric_name}: {reg.baseline_mean:.4f} -> "
                    f"{reg.current_mean:.4f} ({reg.delta_percent:+.1f}%){notes}"
                )
        else:
            lines.append("No regressions detected")

        if self.improvements:
            lines.append("")
            lines.append("Improvements:")
            for imp in self.improvements:
                lines.append(
                    f"  {imp.metric_name}: {imp.baseline_mean:.4f} -> "
                    f"{imp.current_mean:.4f} ({imp.delta_percent:+.1f}%)"
                )

        return "\n".join(lines)


class RegressionDetector:
    """Detects regressions between baseline and current results.

    Uses statistical tests to determine if observed differences
    are significant.

    Example:
        detector = RegressionDetector(significance_level=0.05)
        report = detector.compare(baseline, current_results)

        if report.should_block_ci():
            sys.exit(1)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        min_delta_percent: float = 5.0,
        noise_band_absolute: float = DEFAULT_NOISE_BAND_ABSOLUTE,
        noise_band_aware: bool = True,
    ):
        """Initialize the detector.

        Args:
            significance_level: P-value threshold for significance
            min_delta_percent: Minimum percentage change to consider
            noise_band_absolute: Absolute delta below which a regression
                on a pass-rate-style metric (0-1 scale) is considered
                "within the infra-noise band" when the baseline and
                current infra configs don't match. Defaults to 0.03
                (3pp), following Anthropic's infra-noise study.
            noise_band_aware: If True, compare_with_specs() will mark
                sub-noise-band regressions as ``within_noise_band`` when
                infra configs differ. Set to False to disable the
                downgrade (always treat every delta as real).
        """
        self.significance_level = significance_level
        self.min_delta_percent = min_delta_percent
        self.noise_band_absolute = noise_band_absolute
        self.noise_band_aware = noise_band_aware

    def compare(
        self,
        baseline: TaskBaseline,
        current_results: list[dict[str, Any]],
    ) -> RegressionReport:
        """Compare current results against baseline.

        Args:
            baseline: The baseline to compare against
            current_results: List of result dicts, each with metric values

        Returns:
            RegressionReport with detected regressions
        """
        regressions = []
        improvements = []

        # Collect all metrics from current results
        current_metrics: dict[str, list[float]] = {}
        for result in current_results:
            for metric, value in result.items():
                if isinstance(value, (int, float)):
                    if metric not in current_metrics:
                        current_metrics[metric] = []
                    current_metrics[metric].append(float(value))

        # Compare each metric
        for metric_name, current_values in current_metrics.items():
            metric_baseline = baseline.get_metric(metric_name)

            if not metric_baseline:
                continue

            regression = self._analyze_metric(
                metric_name=metric_name,
                baseline_value=metric_baseline.baseline_value,
                baseline_std=metric_baseline.std_deviation,
                current_values=current_values,
                higher_is_better=metric_baseline.higher_is_better,
            )

            if regression:
                # Determine if it's a regression or improvement
                if metric_baseline.higher_is_better:
                    is_regression = regression.delta < 0
                else:
                    is_regression = regression.delta > 0

                # Threshold downgrade for degenerate samples: when no valid
                # significance test exists, the delta thresholds alone decide
                # — dropping the regression entirely would let a 1.0 -> 0.0
                # pass-rate flip sail through the gate unreported.
                reportable = regression.is_significant or regression.insufficient_data
                if is_regression and reportable:
                    regressions.append(regression)
                elif not is_regression and reportable:
                    improvements.append(regression)

        # Determine overall severity
        if regressions:
            overall_severity = max(r.severity for r in regressions)
        else:
            overall_severity = RegressionSeverity.NONE

        return RegressionReport(
            baseline_id=baseline.task_id,
            baseline_commit=baseline.git_commit,
            has_regression=len(regressions) > 0,
            overall_severity=overall_severity,
            regressions=regressions,
            improvements=improvements,
            summary=self._generate_summary(regressions, improvements),
        )

    def _analyze_metric(
        self,
        metric_name: str,
        baseline_value: float,
        baseline_std: float,
        current_values: list[float],
        higher_is_better: bool,
    ) -> MetricRegression | None:
        """Analyze a single metric for regression."""
        if not current_values:
            return None

        current_mean = float(np.mean(current_values))
        delta = current_mean - baseline_value

        # Calculate percentage change
        if baseline_value != 0:
            delta_percent = (delta / abs(baseline_value)) * 100
        else:
            delta_percent = 100.0 if delta != 0 else 0.0

        # Skip if change is too small
        if abs(delta_percent) < self.min_delta_percent:
            return None

        # Statistical test. p_value stays None when no valid test exists —
        # zero variance on both sides leaves nothing to test against. The
        # regression is then reported on its delta thresholds alone and
        # flagged insufficient_data, never given a fabricated p=0.0.
        p_value: float | None
        if len(current_values) >= 2 and baseline_std > 0:
            # One-sample t-tests are undefined for zero-variance samples;
            # fall back to a z-test against the known baseline spread and
            # avoid scipy's precision-loss RuntimeWarning.
            current_std = float(np.std(current_values, ddof=1))
            if np.isclose(current_std, 0.0, atol=1e-12):
                z = abs(delta) / (baseline_std / np.sqrt(len(current_values)))
                p_value = float(2 * (1 - stats.norm.cdf(z)))
            else:
                _t_stat, p_value_result = stats.ttest_1samp(current_values, baseline_value)
                p_value = float(p_value_result)
        elif len(current_values) >= 2:
            # Can't do proper test without baseline std, use empirical
            current_std = float(np.std(current_values))
            if current_std > 0:
                z = abs(delta) / current_std
                p_value = float(2 * (1 - stats.norm.cdf(z)))
            else:
                p_value = None
        elif baseline_std > 0:
            # Single sample, use z-test with baseline std
            z = abs(delta) / baseline_std
            p_value = float(2 * (1 - stats.norm.cdf(z)))
        else:
            p_value = None

        insufficient_data = p_value is None
        is_significant = p_value is not None and p_value < self.significance_level

        # Determine severity based on percentage change
        abs_pct = abs(delta_percent)
        if abs_pct >= 15:
            severity = RegressionSeverity.SEVERE
        elif abs_pct >= 5:
            severity = RegressionSeverity.MODERATE
        elif abs_pct > 0:
            severity = RegressionSeverity.MINOR
        else:
            severity = RegressionSeverity.NONE

        return MetricRegression(
            metric_name=metric_name,
            baseline_mean=baseline_value,
            current_mean=current_mean,
            delta=delta,
            delta_percent=delta_percent,
            p_value=p_value,
            is_significant=is_significant,
            insufficient_data=insufficient_data,
            severity=severity,
        )

    def _generate_summary(
        self,
        regressions: list[MetricRegression],
        improvements: list[MetricRegression],
    ) -> str:
        """Generate a summary of the analysis."""
        lines = []

        if regressions:
            lines.append(f"REGRESSIONS DETECTED ({len(regressions)} metrics):")
            for r in regressions:
                p_str = (
                    f"p={r.p_value:.4f}"
                    if r.p_value is not None
                    else "p=n/a, insufficient samples"
                )
                lines.append(
                    f"  - {r.metric_name}: {r.baseline_mean:.4f} -> {r.current_mean:.4f} "
                    f"({r.delta_percent:+.1f}%, {p_str}) [{r.severity.value}]"
                )
        else:
            lines.append("No significant regressions detected.")

        if improvements:
            lines.append(f"\nIMPROVEMENTS ({len(improvements)} metrics):")
            for i in improvements:
                lines.append(
                    f"  + {i.metric_name}: {i.baseline_mean:.4f} -> {i.current_mean:.4f} "
                    f"({i.delta_percent:+.1f}%)"
                )

        return "\n".join(lines)

    def compare_multiple(
        self,
        baselines: dict[str, TaskBaseline],
        current_results: dict[str, list[dict[str, Any]]],
    ) -> dict[str, RegressionReport]:
        """Compare multiple tasks against their baselines.

        Args:
            baselines: Dict of task_id -> TaskBaseline
            current_results: Dict of task_id -> list of result dicts

        Returns:
            Dict of task_id -> RegressionReport
        """
        reports = {}

        for task_id, results in current_results.items():
            baseline = baselines.get(task_id)
            if baseline:
                reports[task_id] = self.compare(baseline, results)

        return reports

    def compare_with_specs(
        self,
        baseline: TaskBaseline,
        current_results: list[dict[str, Any]],
        baseline_spec: DecisionSpec | None = None,
        current_spec: DecisionSpec | None = None,
    ) -> RegressionReport:
        """Compare with DecisionSpec awareness for infra-noise detection.

        Wraps ``compare()`` and additionally:

        1. Diffs the two DecisionSpecs' ``infra`` sections and records
           any mismatch in ``report.infra_config_mismatch`` and
           ``report.infra_config_diff``.
        2. For each detected regression, if the **absolute** delta falls
           within ``noise_band_absolute`` (default 3pp) AND the infra
           configs don't match, mark the regression's
           ``within_noise_band`` flag to True. Those regressions still
           show up in the report but are excluded from
           ``blocking_regressions`` so a default ``should_block_ci()``
           call won't gate a merge on ambiguous noise.

        When either spec is omitted, this degrades to ordinary
        ``compare()`` behavior with ``infra_config_mismatch=False``.

        Args:
            baseline: TaskBaseline to compare against.
            current_results: Current run's metric values.
            baseline_spec: DecisionSpec captured when the baseline was
                recorded. Optional but enables infra-noise reasoning.
            current_spec: DecisionSpec for the current run. Optional
                but enables infra-noise reasoning.

        Returns:
            RegressionReport with ``infra_config_mismatch``,
            ``infra_config_diff``, and per-regression
            ``within_noise_band`` annotations populated.
        """
        report = self.compare(baseline, current_results)

        if not self.noise_band_aware or baseline_spec is None or current_spec is None:
            return report

        # Compare the infra sections of the two specs.
        baseline_infra = baseline_spec.infra.to_hash_dict() if baseline_spec.infra else None
        current_infra = current_spec.infra.to_hash_dict() if current_spec.infra else None

        if baseline_infra != current_infra:
            report.infra_config_mismatch = True
            # Record a field-level diff of infra (only fields that changed).
            keys = set((baseline_infra or {}).keys()) | set((current_infra or {}).keys())
            diff: dict[str, tuple[Any, Any]] = {}
            for key in keys:
                b = (baseline_infra or {}).get(key)
                c = (current_infra or {}).get(key)
                if b != c:
                    diff[key] = (b, c)
            report.infra_config_diff = diff

            # Downgrade regressions that fall within the noise band to
            # "within_noise_band" rather than counting as blocking
            # regressions. The absolute delta is what matters here;
            # Anthropic's "3 percentage points" is in absolute units on
            # a 0-1 metric, not a relative percentage.
            for regression in report.regressions:
                if abs(regression.delta) < self.noise_band_absolute:
                    regression.within_noise_band = True

        return report
