"""Regression detection and comparison.

Compares the current trials of a task with its stored baseline, metric by
metric, and reports every drop above the reporting floor together with the
evidence for it. The procedure is specified in ``docs/statistical-contract.md``
("Baseline regression detection"):

- Whether a metric is a 0/1 proportion is decided by the baseline, not by
  where the current sample happens to land: either the baseline declares it
  (``MetricBaseline.is_rate``) or its stored summary is consistent with one
  (mean times ``sample_size`` is a whole count, and any positive recorded
  spread is one 0/1 data of that size could show). Such a metric is
  compared with Boschloo's exact unconditional test on the two counts.
- A continuous metric uses **Welch's** t-test from the stored summary. A
  zero measured spread on one side is not a reason to assume equal
  population variances, so the pooled-variance test is not used. When both
  sides are constant the p-value is the exact permutation value; a single
  current trial against a measured baseline is a prediction-interval t.
- p-values are one-sided in the observed direction and never fabricated. A
  comparison with no valid test is ``insufficient_data``.
- Evidence is never invented for the baseline. Its recorded ``sample_size``
  is used as recorded, and a baseline that stored fewer than two trials has
  no measured spread at all: rather than borrow the current sample's
  evidence to stand in for it, such a comparison usually comes out
  ``undetectable`` and makes the gate unevaluable. It is not powerless --
  one stored passing trial against seven straight failures is p=0.049 on
  the honest counts, and that blocks -- but it cannot decide much else.
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
TEST_PERMUTATION = "exact_permutation"
TEST_PREDICTION = "prediction_t"
# No longer produced: a zero measured spread on one side does not justify
# assuming the two populations share a variance. Kept so findings stored by
# an earlier version still round-trip.
TEST_POOLED = "pooled_t"

# Direction of a one-sided test: "greater" means the baseline mean is higher.
Alternative = Literal["greater", "less"]

_BINARY_TOLERANCE = 1e-9
# ``baseline_value * sample_size`` must be this close to a whole number for
# the stored baseline to be read as a count of successes.
_COUNT_TOLERANCE = 1e-6
# Slack on the widest spread 0/1 data of a given size can show, before a
# recorded spread is judged too large to have come from a proportion.
_BERNOULLI_TOLERANCE = 0.02
# ``trials_needed`` scans stop here; beyond it the advice is "more than".
TRIALS_NEEDED_CAP = 200


def _is_binary(values: Sequence[float]) -> bool:
    return all(
        abs(v) <= _BINARY_TOLERANCE or abs(v - 1.0) <= _BINARY_TOLERANCE for v in values
    )


def _round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def _bernoulli_std(k: int, n: int) -> float:
    """Sample standard deviation (ddof=1) of ``n`` 0/1 values with ``k`` ones."""
    if n < 2:
        return 0.0
    return math.sqrt(k * (n - k) / (n * (n - 1)))


def _is_a_whole_count(mean_b: float, n_b: int) -> bool:
    """Whether ``mean_b`` over ``n_b`` trials describes a count of successes.

    A rate has to sit in [0, 1] and its mean times its size has to be the
    whole number of trials that passed. Half a success over five trials is
    not a proportion, whatever the baseline calls itself: the exact test
    would round it to a count nobody measured, and ``_round_half_up`` breaks
    ties upward, which inflates the baseline and is anti-conservative in the
    blocking direction specifically.
    """
    if n_b < 1 or not math.isfinite(mean_b) or not 0.0 <= mean_b <= 1.0:
        return False
    k = mean_b * n_b
    return abs(k - round(k)) <= _COUNT_TOLERANCE


def _summary_is_a_proportion(mean_b: float, n_b: int, std_b: float | None) -> bool:
    """Whether the stored baseline summary is consistent with a 0/1 rate.

    Every check is on the baseline's own evidence, never on the current
    sample: the mean sits in [0, 1]; the mean times the recorded size is a
    whole count of successes, which is what rules out a continuous score
    such as 0.1 over five trials (half a success is not a count); and any
    positive recorded spread is one 0/1 data of that size could actually
    show. The spread is deliberately only a sanity bound, not a match:
    a real proportion records zero whenever every trial agreed, callers that
    never computed a spread leave the field at its 0.0 default, and
    ``tracelens init`` scaffolds a nominal 0.05 -- none of which is evidence
    against a rate.

    This is an inference for baselines that do not say. A baseline that sets
    ``MetricBaseline.is_rate`` is taken at its word and never reaches here.
    """
    if not _is_a_whole_count(mean_b, n_b):
        return False
    if std_b is None or std_b <= 0.0:
        return True
    # The widest spread 0/1 data of this size can show is at a half-and-half
    # split; anything beyond it did not come from a proportion.
    widest = _bernoulli_std(n_b // 2, n_b)
    return std_b <= widest + _BERNOULLI_TOLERANCE


@lru_cache(maxsize=65536)
def _boschloo_p(k_b: int, n_b: int, k_c: int, n_c: int, alternative: Alternative) -> float:
    """One-sided Boschloo p-value for baseline ``k_b/n_b`` vs current ``k_c/n_c``.

    SciPy's model puts one binomial experiment in each **column** and names
    the two probabilities after the first row's entries, so the baseline and
    the current sample are the two columns, not the two rows. Transposing
    the table asks a different question -- it fixes the wrong margin -- and
    at unequal sizes it answers differently: 7/7 against 2/4 reads p=0.0420
    transposed and p=0.0538 as written here.

    ``alternative="greater"`` therefore tests "the baseline rate is higher"
    (a drop) and ``"less"`` a rise. Cached: a gate over many tasks repeats
    the same small tables.
    """
    table = [[k_b, k_c], [n_b - k_b, n_c - k_c]]
    result = stats.boschloo_exact(table, alternative=alternative)
    p = float(result.pvalue)
    if not math.isfinite(p):
        # Defensive: every column here sums to at least one trial, so SciPy
        # has a defined value for each table this builds. Should that ever
        # change, "no evidence" is the safe reading -- clamping a NaN would
        # silently produce p=0.0 and fabricate significance.
        return 1.0
    return float(min(1.0, max(0.0, p)))


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

    def scaled(self, n_c: int, *, grow_baseline: bool = False) -> _Sides:
        """The same rates and spreads projected onto ``n_c`` current trials.

        Only ``_trials_needed`` uses this, and only to answer "how many
        trials would decide the change we just saw" -- a power projection,
        never a reported p-value. ``grow_baseline`` also re-stores the
        baseline at the same size. A current sample of one has no measured
        spread, so the projection borrows the baseline's; that assumption is
        confined to this planning path.
        """
        n_b = n_c if grow_baseline else self.n_b
        std_c = self.std_c
        if std_c is None and n_c >= 2:
            std_c = self.std_b
        return replace(self, n_b=n_b, n_c=n_c, std_c=std_c)

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
    # A stored size below zero is a corrupt record, not evidence. Floor it
    # rather than handing a negative count to the exact test, which would
    # raise out of SciPy and take the whole run down; the comparison then
    # falls through to "no measured spread" and the gate reports that it
    # could not evaluate this task.
    n_b = max(0, int(metric_baseline.sample_size))
    # Fewer than two stored trials is not evidence of a spread, whatever the
    # field says; and the size is never inflated to match the check, which
    # would credit the baseline with runs it never had.
    declared = n_b < 2
    recorded_std = float(metric_baseline.std_deviation)
    std_b: float | None = None if declared else recorded_std
    # The metric's type comes from the baseline: an explicit declaration
    # when there is one, otherwise whether its stored summary is consistent
    # with a proportion. The current sample can only confirm that, never
    # decide it -- otherwise a continuous score that happens to land on 0
    # changes the test family and the verdict.
    is_rate = metric_baseline.is_rate
    if is_rate is None:
        is_rate = _summary_is_a_proportion(mean_b, n_b, std_b)
    # A declaration says which family the metric belongs to; it cannot supply
    # a count the summary does not contain, nor turn values that are not 0/1
    # into successes. Where either is missing the comparison falls back to
    # the continuous path rather than inventing a table.
    binary = (
        bool(is_rate)
        and _is_a_whole_count(mean_b, n_b)
        and _is_binary(current_values)
    )
    return _Sides(
        binary=binary,
        mean_b=mean_b,
        n_b=n_b,
        std_b=std_b,
        mean_c=mean_c,
        n_c=n_c,
        std_c=std_c,
        baseline_n_assumed=declared,
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
    if not (math.isfinite(sides.mean_b) and math.isfinite(sides.mean_c)):
        # A corrupt baseline value or a non-finite metric. There is nothing
        # to compare, and every branch below would happily return a number
        # for it -- two "constant" sides one of which is NaN reach the
        # permutation value and read as the strongest evidence available.
        return None, None
    if sides.mean_b == sides.mean_c:
        # No difference in either direction. Worth stating before the
        # branches below, because the permutation value is the probability
        # of the most extreme split and only applies when the two sides
        # actually separate -- two constant sides at the SAME value would
        # otherwise read as the strongest evidence the sizes allow.
        return None, 1.0
    std_b, std_c = sides.std_b, sides.std_c
    if std_b is None:
        # The baseline stored fewer than two trials, so it has no measured
        # spread. Standing the current sample's spread in for it would
        # credit the baseline with evidence it never carried, which is how a
        # declared target comes to block a run on its own.
        return None, None
    if std_c is None:
        # One current trial against a measured baseline: the question is
        # whether a single new observation is consistent with the baseline
        # sample, which is a prediction interval, not a two-sample t-test.
        if std_b > 0.0:
            se = std_b * math.sqrt(1.0 + 1.0 / sides.n_b)
            t_stat = (sides.mean_c - sides.mean_b) / se
            df = sides.n_b - 1
            p = (
                float(stats.t.cdf(t_stat, df))
                if alternative == "greater"
                else float(stats.t.sf(t_stat, df))
            )
            if not math.isfinite(p):
                # Clamping would turn a NaN into 0.0 and fabricate
                # significance; "no valid test" is the honest reading.
                return None, None
            return TEST_PREDICTION, min(1.0, max(0.0, p))
        # A constant baseline and one differing observation: the exact
        # permutation value over the pooled values.
        return TEST_PERMUTATION, 1.0 / math.comb(sides.n_b + 1, 1)
    if std_b == 0.0 and std_c == 0.0:
        # Both sides constant: only one split of the pooled values puts all
        # the extreme ones on the current side.
        return TEST_PERMUTATION, 1.0 / math.comb(sides.n_b + sides.n_c, sides.n_c)
    if std_b == 0.0 or std_c == 0.0:
        # Exactly one side showed no variation. That is not a measurement of
        # zero variance, and handing it to Welch as one credits that side's
        # mean with no uncertainty at all: three flat baseline trials against
        # twenty scattered current ones rejected 38 % of unchanged runs, and
        # against fifty, 91 %. Pooling instead fails the other way round --
        # a flat side with the larger n dominates the pooled estimate, which
        # is how a baseline recorded with spread 0 over 100 trials read
        # p=6e-34 against three scattered values. Use the spread that was
        # actually informative for both sides: conservative in either
        # direction, and Welch's degrees of freedom still follow the sizes.
        std_b = std_c = max(std_b, std_c)
    # Welch throughout: a pooled estimate would assume the two populations
    # share a variance, which nothing here shows.
    result = stats.ttest_ind_from_stats(
        sides.mean_b, std_b, sides.n_b,
        sides.mean_c, std_c, sides.n_c,
        equal_var=False, alternative=alternative,
    )
    p = float(result.pvalue)
    if not math.isfinite(p):
        return None, None
    return TEST_WELCH, min(1.0, max(0.0, p))


def _alternative(sides: _Sides) -> Alternative:
    return "greater" if sides.mean_c < sides.mean_b else "less"


def _trials_needed(
    sides: _Sides,
    alpha: float,
    *,
    both_sides: bool = False,
    cap: int = TRIALS_NEEDED_CAP,
) -> int | None:
    """About how many current trials would decide the change just observed.

    Keeps the observed rates and spreads and grows the current side; the
    baseline side grows with it when ``both_sides`` is set (a re-stored
    baseline). ``None`` when the cap is reached first, or when no test
    applies at any size.

    The scan probes upward and then bisects, which assumes significance is
    monotone in the sample size. For a discrete test it is not quite: a
    rounded count can step the wrong way at a particular size, so this is
    the first size the scan lands on rather than provably the smallest. It
    is advice about the order of magnitude to rerun at, not a guarantee.
    """
    if sides.mean_c == sides.mean_b:
        return None
    alternative = _alternative(sides)

    def significant(n: int) -> bool:
        _test, p = _p_value(sides.scaled(n, grow_baseline=both_sides), alternative)
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


# The severity ladder, least to most severe; blocking compares positions.
SEVERITY_ORDER: list[RegressionSeverity] = [
    RegressionSeverity.NONE,
    RegressionSeverity.MINOR,
    RegressionSeverity.MODERATE,
    RegressionSeverity.SEVERE,
]


def severity_at_least(severity: RegressionSeverity, threshold: RegressionSeverity) -> bool:
    """Whether ``severity`` is at or above ``threshold`` on the ladder."""
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(threshold)


def relative_change(delta: float, baseline: float) -> float:
    """``delta`` as a percentage of ``|baseline|`` (100 when the baseline is 0)."""
    if baseline != 0:
        return (delta / abs(baseline)) * 100
    return 100.0 if delta != 0 else 0.0


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
    # ``welch_t``, ``exact_permutation``, ``prediction_t``, or ``None``;
    # ``pooled_t`` only in findings stored by an older version), the two
    # sample sizes as recorded, and the spreads the tests saw
    # (``baseline_std`` is ``None`` when the baseline never measured one).
    # ``baseline_n_assumed`` marks a baseline that stored fewer than two
    # trials: its mean is a declared value rather than a measurement, so it
    # carries no spread and almost no power. The size itself is never
    # inflated to match the check.
    #
    # ``undetectable`` means no effect of any size could have been found at
    # these sample sizes -- for a 0/1 metric because even a total failure
    # would not reject, and for any metric because no valid test exists.
    test: str | None = None
    baseline_n: int | None = None
    current_n: int | None = None
    baseline_n_assumed: bool = False
    baseline_std: float | None = None
    current_std: float | None = None
    higher_is_better: bool = True

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
        return severity_at_least(effective_severity, threshold)

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
        elif self.has_regression:
            # A report a caller built from a summary rather than from
            # findings. It still carries a verdict, and ``should_block_ci``
            # still acts on it, so it must not read as a clean run.
            lines.append(
                f"REGRESSION DETECTED [{self.overall_severity.value.upper()}]"
            )
            if self.summary:
                lines.append("")
                lines.append(f"  {self.summary}")
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
        power_notes: bool = True,
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
            power_notes: Compute ``trials_needed`` / ``undetectable`` on
                every finding ``compare()`` reports. The gate turns this off
                and annotates once more after the run-level correction.
        """
        self.significance_level = significance_level
        self.min_delta_percent = min_delta_percent
        self.noise_band_absolute = noise_band_absolute
        self.noise_band_aware = noise_band_aware
        self.power_notes = power_notes

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
        delta_percent = relative_change(delta, sides.mean_b)

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
            higher_is_better=metric_baseline.higher_is_better,
        )
        self.annotate(finding, self.significance_level, power_notes=self.power_notes)
        return finding

    @staticmethod
    def _sides_of(finding: MetricRegression) -> _Sides | None:
        if finding.baseline_n is None or finding.current_n is None:
            return None
        higher_is_better = finding.higher_is_better
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
        *,
        power_notes: bool = True,
    ) -> None:
        """Set ``is_significant`` and the power notes of a finding.

        ``alpha`` is the level the finding's adjusted p-value (or, without
        a correction, its raw p-value) is held to. ``per_test_level`` is the
        raw level one test must reach to be significant — ``alpha`` itself
        without a correction, ``alpha / m`` under Holm over ``m`` tests —
        and drives ``trials_needed`` and ``undetectable``. With
        ``power_notes=False`` only the significance fields are set.
        """
        level = alpha if per_test_level is None else per_test_level
        p = finding.p_value_adjusted if finding.p_value_adjusted is not None else finding.p_value
        finding.is_significant = p is not None and p <= alpha
        finding.underpowered = not finding.is_significant
        finding.trials_needed = None
        finding.trials_needed_on_both_sides = False
        finding.undetectable = False
        if finding.p_value is None:
            # No valid test exists for this comparison, so no effect of any
            # size could have been detected. That is a fact about the
            # evidence, not a power note, so it is recorded either way.
            finding.undetectable = True
        sides = self._sides_of(finding)
        if not power_notes or sides is None or finding.p_value is None or not finding.underpowered:
            return
        finding.trials_needed = _trials_needed(sides, level)
        if finding.trials_needed is None:
            # Growing the check alone could not decide it, so the baseline is
            # what limits the evidence. This is the advice a baseline of one
            # stored trial needs most, and it used to be skipped for exactly
            # those findings.
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
            # A change far beyond any plausible spread: a t-test then rejects
            # whenever it can, while the exact permutation p-value of two
            # constant sides is fixed by the sizes and may never reach the
            # level.
            scale = max(abs(sides.mean_b), sides.std_b or 0.0, sides.std_c or 0.0, 1.0)
            shift = -1e3 * scale if sides.higher_is_better else 1e3 * scale
            probe = replace(sides, mean_c=sides.mean_b + shift)
            _test, p = _p_value(probe, _alternative(probe))
            return p is not None and p <= level, None
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
