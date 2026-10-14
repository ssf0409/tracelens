"""`relative_delta` when the baseline mean is 0.

A relative change is a fraction of the baseline's magnitude, and a zero mean
has none. `compare_metrics` and `compare_to_baseline_summary` used to report
``+inf`` for a move in either direction: a decline read as an unbounded rise,
and the result was not valid strict JSON. The verdicts use ``delta`` and were
already right; only the reported ratio is unavailable.
"""

import json

import pytest

from tracelens.statistics.inference import (
    ComparisonResult,
    compare_metrics,
    compare_to_baseline_summary,
)

# The keys `ComparisonResult.to_dict()` has always had. A comparison against a
# nonzero mean must keep exactly these, so consumers that compare whole
# dictionaries see no change.
LEGACY_KEYS = {
    "baseline", "current", "delta", "relative_delta", "ci_lower", "ci_upper",
    "confidence", "is_significant", "is_regression", "is_improvement",
    "cohens_d", "effect_magnitude", "p_value",
}

ZERO_MEAN = [-1.0, 1.0, -1.0, 1.0]
UP = [4.0, 5.0, 6.0, 5.0]
DOWN = [-4.0, -5.0, -6.0, -5.0]


def _from_samples(current: list[float], baseline: list[float] = ZERO_MEAN) -> ComparisonResult:
    return compare_metrics(baseline, current, n_bootstrap=200, seed=0)


def _from_summary(current_mean: float, baseline_mean: float = 0.0) -> ComparisonResult:
    return compare_to_baseline_summary(
        baseline_mean=baseline_mean, baseline_std=1.0, baseline_n=10,
        current_mean=current_mean, current_std=1.0, current_n=10,
    )


@pytest.mark.parametrize("current", [UP, DOWN], ids=["up", "down"])
def test_samples_relative_delta_from_zero_mean_is_unavailable(current: list[float]) -> None:
    result = _from_samples(current)

    assert result.relative_delta is None
    assert result.unavailable_fields == ["relative_delta"]
    # The verdict reads the signed delta, which stays available.
    assert result.delta == pytest.approx(sum(current) / len(current))
    assert result.is_significant is True
    assert result.is_regression is (result.delta < 0)
    assert result.is_improvement is (result.delta > 0)


@pytest.mark.parametrize("current_mean", [5.0, -5.0], ids=["up", "down"])
def test_summary_relative_delta_from_zero_mean_is_unavailable(current_mean: float) -> None:
    result = _from_summary(current_mean)

    assert result.relative_delta is None
    assert result.unavailable_fields == ["relative_delta"]
    assert result.delta == current_mean
    assert result.is_significant is True
    assert result.is_regression is (current_mean < 0)
    assert result.is_improvement is (current_mean > 0)


def test_no_movement_from_zero_mean_is_unavailable_too() -> None:
    # 0 / 0 is undefined, not "no relative change": the same rule as the
    # point comparison in `BaselineManager.compare_to_baseline`.
    result = _from_summary(0.0)

    assert result.relative_delta is None
    assert result.unavailable_fields == ["relative_delta"]
    assert result.is_significant is False


@pytest.mark.parametrize(
    "result",
    [_from_samples(DOWN), _from_summary(-5.0)],
    ids=["samples", "summary"],
)
def test_zero_mean_to_dict_names_the_unavailable_field(result: ComparisonResult) -> None:
    row = result.to_dict()

    assert row["relative_delta"] is None
    assert row["unavailable_fields"] == ["relative_delta"]
    assert set(row) == LEGACY_KEYS | {"unavailable_fields"}


@pytest.mark.parametrize(
    "result",
    [_from_samples(UP, baseline=[0.8, 1.2, 0.8, 1.2]), _from_summary(1.5, baseline_mean=1.0)],
    ids=["samples", "summary"],
)
def test_nonzero_mean_to_dict_keeps_its_keys(result: ComparisonResult) -> None:
    row = result.to_dict()

    assert set(row) == LEGACY_KEYS
    assert isinstance(row["relative_delta"], float)
    assert result.unavailable_fields == []


@pytest.mark.parametrize(
    "result",
    [_from_samples(UP), _from_samples(DOWN), _from_summary(5.0), _from_summary(-5.0)],
    ids=["samples-up", "samples-down", "summary-up", "summary-down"],
)
def test_zero_mean_comparison_round_trips_through_strict_json(result: ComparisonResult) -> None:
    row = result.to_dict()

    encoded = json.dumps(row, allow_nan=False)

    assert json.loads(encoded) == row
