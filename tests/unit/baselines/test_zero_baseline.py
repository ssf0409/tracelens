"""`compare_to_baseline` against a metric whose baseline is 0.

A relative change is a fraction of the baseline's magnitude, and a zero
baseline has none. The comparison used to report ``+inf`` for a move in
either direction and fed it to the one-sided relative threshold, so a
higher-is-better metric that fell from 0 passed and a lower-is-better one
that improved from 0 was flagged. Per-policy P&L and per-reason counters
start at 0, so this is the common case, not an edge.
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

    # A relative threshold cannot size a decline from zero, so the decline
    # has no verdict rather than a pass; the improvement is not a regression.
    assert result["pnl"]["regression"] is None
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
    assert "relative_change" in row["unavailable_fields"]
    assert row["delta"] == current


@pytest.mark.parametrize(
    ("higher_is_better", "current", "expected"),
    [
        (True, -50.0, True),   # decline past the absolute limit
        (True, -0.5, False),   # decline within it
        (True, 50.0, False),   # improvement
        (False, 5.0, True),    # worsening past the absolute limit
        (False, 0.5, False),   # worsening within it
        (False, -5.0, False),  # improvement
    ],
    ids=[
        "higher-down-past", "higher-down-within", "higher-up",
        "lower-up-past", "lower-up-within", "lower-down",
    ],
)
def test_absolute_threshold_alone_decides_from_zero(
    tmp_path: Path, higher_is_better: bool, current: float, expected: bool
) -> None:
    row = _compare(
        tmp_path,
        current,
        higher_is_better=higher_is_better,
        absolute_threshold=-1.0,
        relative_threshold=0.1,
    )

    assert row["regression"] is expected


@pytest.mark.parametrize(
    ("higher_is_better", "current", "expected"),
    [
        (True, -50.0, None),   # worsening: nothing configured can size it
        (True, 50.0, False),   # improvement: never a regression
        (False, 5.0, None),
        (False, -5.0, False),
        (True, 0.0, False),    # no movement
        (False, 0.0, False),
    ],
    ids=["higher-down", "higher-up", "lower-up", "lower-down", "higher-flat", "lower-flat"],
)
def test_relative_only_threshold_from_zero_judges_direction_only(
    tmp_path: Path, higher_is_better: bool, current: float, expected: bool | None
) -> None:
    row = _compare(tmp_path, current, higher_is_better=higher_is_better, relative_threshold=0.1)

    assert row["regression"] is expected
    assert ("regression" in row["unavailable_fields"]) is (expected is None)


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
