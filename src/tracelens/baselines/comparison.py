"""Regression detection and comparison.

Compares the current trials of a task with its stored baseline, metric by
metric, and reports every drop above the reporting floor together with the
evidence for it. The procedure is specified in ``docs/statistical-contract.md``
("Baseline regression detection"):

- A 0/1-valued metric (``pass_rate``, or any metric whose current samples
  are all 0 or 1) is compared with Boschloo's exact unconditional test on
  the two success counts. The baseline count is its mean times its
  ``sample_size``.
- A continuous metric uses a two-sample t-test from the stored summary:
  Welch's when both sides have a measured spread, pooled-variance when one
  side has none; when both sides are constant, the exact permutation
  p-value. A single current trial is compared with the baseline sample as a
  prediction interval.
- p-values are one-sided in the observed direction and never fabricated. A
  comparison with no valid test is ``insufficient_data``.
- A baseline whose ``sample_size`` was never recorded (the default, 1) is
  treated as if it had been measured with as many trials as the current
  check (``baseline_n_assumed``); a baseline spread that was never recorded
  is taken from the current sample.
- Blocking needs both an effect at or above the severity threshold and a
  significant test. A drop that is reported but not significant is
  ``underpowered`` and carries the number of trials that would decide it;
  a comparison whose sizes cannot reject even a total failure is
  ``undetectable``.

Multiplicity across the tasks of one run belongs to the gate
(:func:`tracelens.reporting.gate.evaluate_gate`), which Holm-adjusts the
per-task p-values with :func:`holm_adjusted`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats

from tracelens.baselines.manager import MetricBaseline, TaskBaseline
from tracelens.core._time import utc_now
from tracelens.core.decision_spec import DecisionSpec

# Default noise band in absolute metric units — i.e. 0.03 means "a score
# change of 3 percentage points on a 0-1 metric." Comes from Anthropic's
# "Quantifying infrastructure noise in agentic coding evals" (Feb 2026):
# "Until resource methodology is standardized, our data suggests that
# leaderboard differences below 3 percentage points deserve skepticism
# until the eval configuration is documented and matched."
DEFAULT_NOISE_BAND_ABSOLUTE: float = 0.03

# Per-comparison significance level; the gate applies it after Holm.
DEFAULT_SIGNIFICANCE_LEVEL: float = 0.05

# Names recorded on ``MetricRegression.test``.
TEST_BOSCHLOO = "boschloo_exact"
TEST_WELCH = "welch_t"
TEST_POOLED = "pooled_t"
TEST_PERMUTATION = "exact_permutation"

# Direction of a one-sided test: "greater" means the baseline mean is higher.
Alternative = Literal["greater", "less"]

_BINARY_TOLERANCE = 1e-9
# ``trials_needed`` scans stop here; beyond it the advice is "more than".
TRIALS_NEEDED_CAP = 200


def _is_binary(values: Sequence[float]) -> bool:
    return all(
        abs(v) <= _BINARY_TOLERANCE or abs(v - 1.0) <= _BINARY_TOLERANCE for v in values
    )


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


@lru_cache(maxsize=65536)
def _boschloo_p(k_b: int, n_b: int, k_c: int, n_c: int, alternative: Alternative) -> float:
    """One-sided Boschloo p-value for baseline ``k_b/n_b`` vs current ``k_c/n_c``.

    ``alternative="greater"`` tests "the baseline rate is higher" (a drop);
    ``"less"`` tests a rise. Cached: a gate over many tasks repeats the same
    small tables.
    """
    table = [[k_b, n_b - k_b], [k_c, n_c - k_c]]
    result = stats.boschloo_exact(table, alternative=alternative)
    return float(min(1.0, max(0.0, float(result.pvalue))))


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order.

    Rejecting every hypothesis whose adjusted p-value is at or below alpha
    keeps the family-wise error rate at or below alpha.
    """
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


@dataclass(frozen=True)
class _Sides:
    """The two sides of one metric comparison as the tests see them."""

    binary: bool
    mean_b: float
    n_b: int
    std_b: float | None  # None: never recorded
    mean_c: float
    n_c: int
    std_c: float | None  # None: undefined (n_c < 2)
    baseline_n_assumed: bool
    higher_is_better: bool

    @property
    def k_b(self) -> int:
        return min(self.n_b, max(0, _round_half_up(self.mean_b * self.n_b)))

    @property
    def k_c(self) -> int:
        return min(self.n_c, max(0, _round_half_up(self.mean_c * self.n_c)))

    def scaled(self, n_c: int) -> _Sides:
        """The same rates and spreads observed with ``n_c`` current trials."""
        n_b = n_c if self.baseline_n_assumed else self.n_b
        return replace(self, n_b=n_b, n_c=n_c)

    def worst_case(self) -> _Sides:
        """Every current trial at the bad end of a 0/1 metric."""
        return replace(self, mean_c=0.0 if self.higher_is_better else 1.0)


def _sides(
    metric_baseline: MetricBaseline, current_values: Sequence[float]
) -> _Sides:
    n_c = len(current_values)
    mean_c = float(np.mean(current_values))
    std_c = float(np.std(current_values, ddof=1)) if n_c >= 2 else None
    mean_b = float(metric_baseline.baseline_value)
    assumed = metric_baseline.sample_size < 2
    n_b = n_c if assumed else int(metric_baseline.sample_size)
    std_b: float | None = float(metric_baseline.std_deviation)
    if assumed and metric_baseline.std_deviation <= 0.0:
        std_b = None  # never measured: the current spread stands in
    binary = _is_binary(current_values) and 0.0 <= mean_b <= 1.0
    return _Sides(
        binary=binary,
        mean_b=mean_b,
        n_b=n_b,
        std_b=std_b,
        mean_c=mean_c,
        n_c=n_c,
        std_c=std_c,
        baseline_n_assumed=assumed,
        higher_is_better=metric_baseline.higher_is_better,
    )


def _p_value(sides: _Sides, alternative: Alternative) -> tuple[str | None, float | None]:
    """The test that applies to ``sides`` and its one-sided p-value.

    ``alternative`` is ``"greater"`` when the baseline mean is the larger
    one (the observed change is a drop) and ``"less"`` otherwise.
    """
    if sides.binary:
        return TEST_BOSCHLOO, _boschloo_p(
            sides.k_b, sides.n_b, sides.k_c, sides.n_c, alternative
        )
    std_b = sides.std_b
    std_c = sides.std_c
    if std_b is None:
        std_b = std_c  # unrecorded baseline spread: assume the current one
    if std_b is None:
        return None, None  # one trial against a declared number: no test
    if sides.n_c >= 2 and std_c is not None and std_b > 0.0 and std_c > 0.0:
        test, equal_var = TEST_WELCH, False
    elif std_b > 0.0 or (std_c is not None and std_c > 0.0):
        test, equal_var = TEST_POOLED, True
    else:
        # Both sides constant: only one split of the pooled values puts all
        # the extreme ones on the current side.
        return TEST_PERMUTATION, 1.0 / math.comb(sides.n_b + sides.n_c, sides.n_c)
    if sides.n_b + sides.n_c < 3:
        return None, None
    result = stats.ttest_ind_from_stats(
        sides.mean_b, std_b, sides.n_b,
        sides.mean_c, std_c if std_c is not None else 0.0, sides.n_c,
        equal_var=equal_var, alternative=alternative,
    )
    p = float(result.pvalue)
    if math.isnan(p):
        return None, None
    return test, min(1.0, max(0.0, p))


def _alternative(sides: _Sides) -> Alternative:
    return "greater" if sides.mean_c < sides.mean_b else "less"


def _trials_needed(
    sides: _Sides,
    alpha: float,
    *,
    both_sides: bool = False,
    cap: int = TRIALS_NEEDED_CAP,
) -> int | None:
    """Smallest current sample size at which these sides would be significant.

    Keeps the observed rates and spreads and grows the current side; the
    baseline side grows with it when its size was assumed or when
    ``both_sides`` is set (a re-stored baseline). ``None`` when the cap is
    reached first, or when no test applies at any size.
    """
    if sides.mean_c == sides.mean_b:
        return None
    if both_sides:
        sides = replace(sides, baseline_n_assumed=True)
    alternative = _alternative(sides)

    def significant(n: int) -> bool:
        _test, p = _p_value(sides.scaled(n), alternative)
        return p is not None and p <= alpha

    low = sides.n_c  # known not significant (or the caller would not ask)
    high = low + 1
    while high <= cap and not significant(high):
        low = high
        high = max(high + 1, int(high * 1.5))
    if high > cap:
        return None
    while high - low > 1:  # bisect the first size that decides it
        mid = (low + high) // 2
        if significant(mid):
            high = mid
        else:
            low = mid
    return high


class RegressionSeverity(str, Enum):
    """Severity levels for regressions."""

    NONE = "none"           # No regression
    MINOR = "minor"         # < 5% decline
    MODERATE = "moderate"   # 5-15% decline (default blocking threshold)
    SEVERE = "severe"       # > 15% decline


def severity_for(delta_percent: float) -> RegressionSeverity:
    """Severity from the size of a relative change alone."""
    abs_pct = abs(delta_percent)
    if abs_pct >= 15:
        return RegressionSeverity.SEVERE
    if abs_pct >= 5:
        return RegressionSeverity.MODERATE
    if abs_pct > 0:
        return RegressionSeverity.MINOR
    return RegressionSeverity.NONE


class MetricRegression(BaseModel):
    """One observed change in one metric, with the evidence for it."""

    metric_name: str
    baseline_mean: float
    current_mean: float
    delta: float
    delta_percent: float

    # Statistical evidence. ``p_value`` is one-sided in the observed
    # direction and ``None`` when no valid test exists — never a fabricated
    # 0.0; such findings carry ``insufficient_data=True``. ``is_significant``
    # compares ``p_value_adjusted`` (set by the gate after Holm) or, without
    # a correction, ``p_value`` with the significance level.
    p_value: float | None
    is_significant: bool
    insufficient_data: bool = False
    p_value_adjusted: float | None = None

    # Severity is derived from ``delta_percent`` alone and reported next to
    # significance; blocking needs both.
    severity: RegressionSeverity

    # Which tasks were affected
    affected_tasks: list[str] = Field(default_factory=list)

    # Noise-awareness: True if the absolute delta falls within the noise
    # band AND the baseline/current infra configurations don't match.
    # Deltas flagged this way are surfaced but do NOT block CI by default
    # (see RegressionReport.blocking_regressions).
    within_noise_band: bool = False

    # How the evidence was produced: the test (``boschloo_exact``,
    # ``welch_t``, ``pooled_t``, ``exact_permutation``, or ``None``), the two
    # sample sizes, whether the baseline size was assumed equal to the
    # current one, and the spreads the t-tests saw.
    test: str | None = None
    baseline_n: int | None = None
    current_n: int | None = None
    baseline_n_assumed: bool = False
    baseline_std: float | None = None
    current_std: float | None = None

    # A change that is reported but not significant is underpowered;
    # ``trials_needed`` is about how many trials would decide the observed
    # rates (``None`` past the cap or without a test): current trials
    # against the stored baseline, or, when the stored baseline is too
    # small for any number of current trials to decide it, trials on each
    # side (``trials_needed_on_both_sides``). A 0/1 comparison whose sizes
    # cannot reject even a total failure is ``undetectable``.
    underpowered: bool = False
    trials_needed: int | None = None
    trials_needed_on_both_sides: bool = False
    undetectable: bool = False

    def evidence_text(self) -> str:
        """Short human-readable evidence note for reports."""
        if self.p_value is None:
            return "no valid test (insufficient data)"
        p_text = f"p={self.p_value:.4f}"
        if self.p_value_adjusted is not None and self.p_value_adjusted != self.p_value:
            p_text += f" (adjusted {self.p_value_adjusted:.4f})"
        if self.is_significant:
            return f"{p_text}, significant"
        text = f"{p_text}, not significant"
        if self.trials_needed is None:
            text += f"; more than {TRIALS_NEEDED_CAP} trials would be needed"
        elif self.trials_needed_on_both_sides:
            text += (
                f"; about {self.trials_needed} trials on each side would decide it "
                f"(the baseline's {self.baseline_n} limit the evidence; re-store it "
                "from more runs)"
            )
        else:
            text += f"; about {self.trials_needed} current trials would decide it"
        return text


class RegressionReport(BaseModel):
    """Complete regression analysis report."""

    baseline_id: str | None = None
    baseline_commit: str | None = None
    current_commit: str | None = None

    # Overall assessment. ``has_regression`` is True when a significant drop
    # was observed (whether or not it blocks); ``overall_severity`` is the
    # worst blocking regression, so it agrees with ``should_block_ci``.
    has_regression: bool = False
    overall_severity: RegressionSeverity = RegressionSeverity.NONE

    # Every reported drop, significant or not (each carries its evidence)
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
    def significant_regressions(self) -> list[MetricRegression]:
        """Regressions whose test rejected (noise-flagged ones included)."""
        return [r for r in self.regressions if r.is_significant]

    @property
    def underpowered_regressions(self) -> list[MetricRegression]:
        """Reported drops the evidence could not confirm."""
        return [r for r in self.regressions if not r.is_significant]

    @property
    def blocking_regressions(self) -> list[MetricRegression]:
        """Regressions that should actually block CI.

        Significant regressions that are not marked ``within_noise_band`` —
        those are within the infra-noise uncertainty and shouldn't gate
        merges until the eval configuration is matched. A drop that is
        reported but not significant never blocks.
        """
        return [r for r in self.regressions if r.is_significant and not r.within_noise_band]

    def recompute(self) -> None:
        """Refresh ``has_regression`` and ``overall_severity`` from the findings.

        Call after changing a finding's significance (the gate does, after
        Holm adjustment).
        """
        self.has_regression = bool(self.significant_regressions)
        self.overall_severity = max(
            (r.severity for r in self.blocking_regressions),
            default=RegressionSeverity.NONE,
        )

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
                every significant regression as blocking regardless of noise.

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
        # the significant findings — filtered to blocking_regressions on the
        # lenient path, unfiltered on the strict path — so the stored
        # overall_severity can't defeat ignore_noise_band=False and an
        # underpowered drop can never block. When the list is empty (e.g. a
        # hand-constructed report where the caller only set
        # overall_severity), fall back to the declared severity so existing
        # callers keep working.
        if self.regressions:
            considered = (
                self.blocking_regressions if ignore_noise_band else self.significant_regressions
            )
            effective_severity = max(
                (r.severity for r in considered),
                default=RegressionSeverity.NONE,
            )
        else:
            effective_severity = self.overall_severity
        return severity_order.index(effective_severity) >= severity_order.index(threshold)

    def to_ci_output(self) -> str:
        """Generate CI-friendly output."""
        lines = []

        if self.regressions:
            if self.has_regression:
                lines.append(
                    f"REGRESSION DETECTED [{self.overall_severity.value.upper()}]"
                )
            else:
                lines.append(
                    f"No significant regression ({len(self.regressions)} observed "
                    "drop(s) not significant at these sample sizes)"
                )
            lines.append("")
            for reg in self.regressions:
                notes = f" [{reg.evidence_text()}]"
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
                    f"{imp.current_mean:.4f} ({imp.delta_percent:+.1f}%) "
                    f"[{imp.evidence_text()}]"
                )

        return "\n".join(lines)


class RegressionDetector:
    """Detects regressions between baseline and current results.

    Uses exact and t-tests on the stored baseline summary and the current
    per-trial samples to say how strong the evidence for each change is.

    Example:
        detector = RegressionDetector(significance_level=0.05)
        report = detector.compare(baseline, current_results)

        if report.should_block_ci():
            sys.exit(1)
    """

    def __init__(
        self,
        significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
        min_delta_percent: float = 5.0,
        noise_band_absolute: float = DEFAULT_NOISE_BAND_ABSOLUTE,
        noise_band_aware: bool = True,
    ):
        """Initialize the detector.

        Args:
            significance_level: A finding is significant when its one-sided
                p-value is at or below this level.
            min_delta_percent: Minimum relative change to report at all.
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

        Every metric change at or above ``min_delta_percent`` is reported
        with its evidence; ``should_block_ci`` decides from the significant
        ones.

        Args:
            baseline: The baseline to compare against
            current_results: List of result dicts, each with metric values
                (one dict per trial)

        Returns:
            RegressionReport with observed regressions and improvements
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

            finding = self._analyze_metric(metric_name, metric_baseline, current_values)
            if finding is None:
                continue
            if metric_baseline.higher_is_better:
                is_regression = finding.delta < 0
            else:
                is_regression = finding.delta > 0
            if is_regression:
                regressions.append(finding)
            else:
                improvements.append(finding)

        report = RegressionReport(
            baseline_id=baseline.task_id,
            baseline_commit=baseline.git_commit,
            regressions=regressions,
            improvements=improvements,
        )
        report.recompute()
        report.summary = self._generate_summary(regressions, improvements)
        return report

    def _analyze_metric(
        self,
        metric_name: str,
        metric_baseline: MetricBaseline,
        current_values: Sequence[float],
    ) -> MetricRegression | None:
        """Analyze a single metric: effect size, test, evidence notes."""
        if not current_values:
            return None
        sides = _sides(metric_baseline, current_values)
        delta = sides.mean_c - sides.mean_b

        # Calculate percentage change
        if sides.mean_b != 0:
            delta_percent = (delta / abs(sides.mean_b)) * 100
        else:
            delta_percent = 100.0 if delta != 0 else 0.0

        # Skip if change is too small
        if abs(delta_percent) < self.min_delta_percent:
            return None

        test, p_value = _p_value(sides, _alternative(sides))
        finding = MetricRegression(
            metric_name=metric_name,
            baseline_mean=sides.mean_b,
            current_mean=sides.mean_c,
            delta=delta,
            delta_percent=delta_percent,
            p_value=p_value,
            is_significant=p_value is not None and p_value <= self.significance_level,
            insufficient_data=p_value is None,
            severity=severity_for(delta_percent),
            test=test,
            baseline_n=sides.n_b,
            current_n=sides.n_c,
            baseline_n_assumed=sides.baseline_n_assumed,
            baseline_std=sides.std_b,
            current_std=sides.std_c,
        )
        self.annotate(finding, self.significance_level)
        return finding

    @staticmethod
    def _sides_of(finding: MetricRegression, higher_is_better: bool = True) -> _Sides | None:
        if finding.baseline_n is None or finding.current_n is None:
            return None
        return _Sides(
            binary=finding.test == TEST_BOSCHLOO,
            mean_b=finding.baseline_mean,
            n_b=finding.baseline_n,
            std_b=finding.baseline_std,
            mean_c=finding.current_mean,
            n_c=finding.current_n,
            std_c=finding.current_std,
            baseline_n_assumed=finding.baseline_n_assumed,
            higher_is_better=higher_is_better,
        )

    def annotate(
        self,
        finding: MetricRegression,
        alpha: float,
        per_test_level: float | None = None,
    ) -> None:
        """Set ``is_significant`` and the power notes of a finding.

        ``alpha`` is the level the finding's adjusted p-value (or, without
        a correction, its raw p-value) is held to. ``per_test_level`` is the
        raw level one test must reach to be significant — ``alpha`` itself
        without a correction, ``alpha / m`` under Holm over ``m`` tests —
        and drives ``trials_needed`` and ``undetectable``.
        """
        level = alpha if per_test_level is None else per_test_level
        p = finding.p_value_adjusted if finding.p_value_adjusted is not None else finding.p_value
        finding.is_significant = p is not None and p <= alpha
        finding.underpowered = not finding.is_significant
        finding.trials_needed = None
        finding.trials_needed_on_both_sides = False
        finding.undetectable = False
        sides = self._sides_of(finding)
        if sides is None or finding.p_value is None or not finding.underpowered:
            return
        finding.trials_needed = _trials_needed(sides, level)
        if finding.trials_needed is None and not sides.baseline_n_assumed:
            finding.trials_needed = _trials_needed(sides, level, both_sides=True)
            finding.trials_needed_on_both_sides = finding.trials_needed is not None
        if sides.binary:
            worst = sides.worst_case()
            _test, worst_p = _p_value(worst, _alternative(worst))
            finding.undetectable = worst_p is not None and worst_p > level

    def detectability(
        self,
        metric_baseline: MetricBaseline,
        current_values: Sequence[float],
        level: float,
    ) -> tuple[bool, int | None]:
        """Whether a total failure would be significant at these sample sizes.

        Returns ``(detectable, trials_needed)``. For a 0/1 metric,
        ``trials_needed`` is the current sample size at which a total
        failure would reach ``level`` (``None`` past the cap). A continuous
        metric with a valid test is detectable for a large enough effect;
        one with no test at all is not.
        """
        if not current_values:
            return False, None
        sides = _sides(metric_baseline, current_values)
        if not sides.binary:
            probe = replace(sides, mean_c=sides.mean_b - 1.0)
            _test, p = _p_value(probe, _alternative(probe))
            return p is not None, None
        worst = sides.worst_case()
        _test, p = _p_value(worst, _alternative(worst))
        if p is not None and p <= level:
            return True, None
        return False, _trials_needed(worst, level)

    def _generate_summary(
        self,
        regressions: list[MetricRegression],
        improvements: list[MetricRegression],
    ) -> str:
        """Generate a summary of the analysis."""
        lines = []

        significant = [r for r in regressions if r.is_significant]
        if significant:
            lines.append(f"REGRESSIONS DETECTED ({len(significant)} metrics):")
        elif regressions:
            lines.append(
                f"No significant regressions ({len(regressions)} observed drop(s) "
                "not significant):"
            )
        else:
            lines.append("No regressions detected.")
        for r in regressions:
            lines.append(
                f"  - {r.metric_name}: {r.baseline_mean:.4f} -> {r.current_mean:.4f} "
                f"({r.delta_percent:+.1f}%, {r.evidence_text()}) [{r.severity.value}]"
            )

        if improvements:
            lines.append(f"\nIMPROVEMENTS ({len(improvements)} metrics):")
            for i in improvements:
                lines.append(
                    f"  + {i.metric_name}: {i.baseline_mean:.4f} -> {i.current_mean:.4f} "
                    f"({i.delta_percent:+.1f}%, {i.evidence_text()})"
                )

        return "\n".join(lines)

    def compare_multiple(
        self,
        baselines: dict[str, TaskBaseline],
        current_results: dict[str, list[dict[str, Any]]],
    ) -> dict[str, RegressionReport]:
        """Compare multiple tasks against their baselines.

        Each report is uncorrected; apply :func:`holm_adjusted` across the
        tasks (as the gate does) when one decision covers all of them.

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
        3. Recomputes ``overall_severity`` from the remaining blocking
           regressions — a report whose regressions are all noise-flagged
           reads ``NONE`` while ``has_regression`` stays True — and
           appends a note about the noise-flagged count to ``summary``.

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
            ``within_noise_band`` annotations populated, and
            ``overall_severity`` recomputed from the blocking regressions.
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

            # Keep overall_severity consistent with the blocking decision:
            # a noise-only report must not read SEVERE while
            # should_block_ci() returns False. has_regression stays True —
            # the regression was observed and is surfaced, just not
            # blocking.
            report.recompute()
            noise_flagged = sum(1 for r in report.regressions if r.within_noise_band)
            if noise_flagged:
                report.summary += (
                    f"\nNOTE: {noise_flagged} regression(s) fall within the "
                    f"{self.noise_band_absolute} infra-noise band under a "
                    "mismatched infra config — surfaced but not blocking."
                )

        return report
