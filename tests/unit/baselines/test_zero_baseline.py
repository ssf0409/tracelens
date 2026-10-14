"""`compare_to_baseline` against a metric whose baseline is 0.

A relative change is a fraction of the baseline's magnitude, and a zero
baseline has none, so the number is unavailable. A relative bound on it is
not: ``r * |0| = 0`` allows no movement, so any move in the worse direction
exceeds every relative threshold, the same verdict a baseline of 1e-300
gives. The comparison used to report ``+inf`` for a move in either direction
and feed it to the one-sided threshold, so a higher-is-better metric that
fell from 0 passed and a lower-is-better one that improved from 0 was
flagged. Per-policy P&L and per-reason counters start at 0, so this is the
common case, not an edge.
"""

import json
from pathlib import Path

import pytest

from tracelens.baselines.manager import BaselineManager, TaskBaseline


def _compare(
    tmp_path: Path,
    current: float,
    *,
    higher_is_better: bool,
    absolute_threshold: float | None = None,
    relative_threshold: float | None = None,
    baseline_value: float = 0.0,
) -> dict:
    manager = BaselineManager(tmp_path / "baselines.json")
    baseline = TaskBaseline(task_id="t")
    baseline.add_metric(
        "m",
        baseline_value,
        absolute_threshold=absolute_threshold,
        relative_threshold=relative_threshold,
        higher_is_better=higher_is_better,
    )
    manager.set_baseline(baseline)
    return manager.compare_to_baseline("t", {"m": current})["m"]


def test_reported_case_flags_the_decline_and_not_the_improvement(tmp_path: Path) -> None:
    manager = BaselineManager(tmp_path / "baselines.json")
    baseline = TaskBaseline(task_id="t")
    baseline.add_metric("pnl", 0.0, relative_threshold=0.1, higher_is_better=True)
    baseline.add_metric("loss", 0.0, relative_threshold=0.1, higher_is_better=False)
    manager.set_baseline(baseline)

    result = manager.compare_to_baseline("t", {"pnl": -50.0, "loss": -5.0})

    assert result["pnl"]["regression"] is True
    assert result["loss"]["regression"] is False


@pytest.mark.parametrize(
    ("higher_is_better", "current"),
    [(True, 50.0), (True, -50.0), (False, 5.0), (False, -5.0)],
    ids=["higher-up", "higher-down", "lower-up", "lower-down"],
)
def test_relative_change_from_zero_is_unavailable(
    tmp_path: Path, higher_is_better: bool, current: float
) -> None:
    row = _compare(tmp_path, current, higher_is_better=higher_is_better, relative_threshold=0.1)

    assert row["relative_change"] is None
    assert row["unavailable_fields"] == ["relative_change"]
    assert row["delta"] == current


@pytest.mark.parametrize(
    ("higher_is_better", "current", "expected"),
    [
        (True, -50.0, True),   # worsening: exceeds a bound that allows none
        (True, -0.01, True),
        (True, 50.0, False),   # improvement: never a regression
        (False, 5.0, True),
        (False, 0.01, True),
        (False, -5.0, False),
        (True, 0.0, False),    # no movement
        (False, 0.0, False),
    ],
    ids=[
        "higher-down", "higher-down-small", "higher-up",
        "lower-up", "lower-up-small", "lower-down", "higher-flat", "lower-flat",
    ],
)
def test_relative_threshold_from_zero_allows_no_worsening(
    tmp_path: Path, higher_is_better: bool, current: float, expected: bool
) -> None:
    row = _compare(tmp_path, current, higher_is_better=higher_is_better, relative_threshold=0.1)

    assert row["regression"] is expected


@pytest.mark.parametrize("higher_is_better", [True, False])
def test_zero_baseline_verdict_matches_a_vanishing_one(
    tmp_path: Path, higher_is_better: bool
) -> None:
    worse = -0.5 if higher_is_better else 0.5
    verdicts = [
        _compare(
            tmp_path, worse, higher_is_better=higher_is_better,
            relative_threshold=0.1, baseline_value=baseline,
        )["regression"]
        for baseline in (0.0, 1e-300)
    ]

    assert verdicts == [True, True]


@pytest.mark.parametrize(
    ("higher_is_better", "current", "relative_threshold", "expected"),
    [
        (True, -50.0, None, True),    # past the absolute limit
        (True, -0.5, None, False),    # within it, nothing else set
        (True, -0.5, 0.1, True),      # within it, but the relative bound allows none
        (True, 50.0, 0.1, False),     # improvement
        (False, 5.0, None, True),
        (False, 0.5, None, False),
        (False, 0.5, 0.1, True),
        (False, -5.0, 0.1, False),
    ],
    ids=[
        "higher-down-past", "higher-down-within", "higher-down-within-relative",
        "higher-up", "lower-up-past", "lower-up-within", "lower-up-within-relative",
        "lower-down",
    ],
)
def test_either_threshold_flags_a_worsening_from_zero(
    tmp_path: Path,
    higher_is_better: bool,
    current: float,
    relative_threshold: float | None,
    expected: bool,
) -> None:
    row = _compare(
        tmp_path,
        current,
        higher_is_better=higher_is_better,
        absolute_threshold=-1.0,
        relative_threshold=relative_threshold,
    )

    assert row["regression"] is expected


def test_nonzero_baseline_row_keeps_its_shape(tmp_path: Path) -> None:
    row = _compare(
        tmp_path, 0.85, higher_is_better=True, relative_threshold=0.1, baseline_value=1.0
    )

    assert row == {
        "baseline": 1.0, "current": 0.85, "delta": pytest.approx(-0.15),
        "relative_change": pytest.approx(-0.15), "regression": True,
        "z_score": 0.0, "higher_is_better": True,
    }


def test_zero_baseline_comparison_round_trips_through_strict_json(tmp_path: Path) -> None:
    manager = BaselineManager(tmp_path / "baselines.json")
    baseline = TaskBaseline(task_id="t")
    for name, higher_is_better in (("pnl", True), ("loss", False)):
        baseline.add_metric(
            name, 0.0, relative_threshold=0.1, higher_is_better=higher_is_better
        )
    manager.set_baseline(baseline)
    result = manager.compare_to_baseline("t", {"pnl": -50.0, "loss": 5.0})

    encoded = json.dumps(result, allow_nan=False)

    assert json.loads(encoded) == result
