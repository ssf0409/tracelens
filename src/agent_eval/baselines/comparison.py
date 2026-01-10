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

from agent_eval.baselines.manager import TaskBaseline


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

    # Statistical significance
    p_value: float
    is_significant: bool

    severity: RegressionSeverity

    # Which tasks were affected
    affected_tasks: list[str] = Field(default_factory=list)


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
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def should_block_ci(
        self,
        threshold: RegressionSeverity = RegressionSeverity.MODERATE,
    ) -> bool:
        """Determine if CI should be blocked based on severity.

        Args:
            threshold: Minimum severity to block. Default: MODERATE

        Returns:
            True if CI should be blocked
        """
        severity_order = [
            RegressionSeverity.NONE,
            RegressionSeverity.MINOR,
            RegressionSeverity.MODERATE,
            RegressionSeverity.SEVERE,
        ]
        return severity_order.index(self.overall_severity) >= severity_order.index(threshold)

    def to_ci_output(self) -> str:
        """Generate CI-friendly output."""
        lines = []

        if self.has_regression:
            lines.append(f"REGRESSION DETECTED [{self.overall_severity.value.upper()}]")
            lines.append("")

            for reg in self.regressions:
                lines.append(
                    f"  {reg.metric_name}: {reg.baseline_mean:.4f} -> "
                    f"{reg.current_mean:.4f} ({reg.delta_percent:+.1f}%)"
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
    ):
        """Initialize the detector.

        Args:
            significance_level: P-value threshold for significance
            min_delta_percent: Minimum percentage change to consider
        """
        self.significance_level = significance_level
        self.min_delta_percent = min_delta_percent

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

                if is_regression and regression.is_significant:
                    regressions.append(regression)
                elif not is_regression and regression.is_significant:
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

        # Statistical test
        if len(current_values) >= 2 and baseline_std > 0:
            # One-sample t-test against baseline
            t_stat, p_value = stats.ttest_1samp(current_values, baseline_value)
            p_value = float(p_value)
        elif len(current_values) >= 2:
            # Can't do proper test without baseline std, use empirical
            current_std = float(np.std(current_values))
            if current_std > 0:
                z = abs(delta) / current_std
                p_value = 2 * (1 - stats.norm.cdf(z))
            else:
                p_value = 0.0 if delta != 0 else 1.0
        else:
            # Single sample, use z-test with baseline std
            if baseline_std > 0:
                z = abs(delta) / baseline_std
                p_value = 2 * (1 - stats.norm.cdf(z))
            else:
                p_value = 0.0 if delta != 0 else 1.0

        is_significant = p_value < self.significance_level

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
                lines.append(
                    f"  - {r.metric_name}: {r.baseline_mean:.4f} -> {r.current_mean:.4f} "
                    f"({r.delta_percent:+.1f}%, p={r.p_value:.4f}) [{r.severity.value}]"
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
