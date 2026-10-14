"""`compare_to_baseline_with_ci` against a metric whose baseline is 0.

The CI comparison passed through the inference layer's ``relative_delta``,
which was ``+inf`` for a move off a zero mean in either direction. Its
verdict (`is_regression`) reads the signed ``delta`` and the interval, so
only the reported ratio changes: it is unavailable, and the row says so.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from tracelens.baselines.manager import BaselineManager, TaskBaseline

UP = [4.0, 5.0, 6.0, 5.0]
DOWN = [-4.0, -5.0, -6.0, -5.0]


def _manager(tmp_path: Path, metrics: dict[str, tuple[float, bool]]) -> BaselineManager:
    manager = BaselineManager(tmp_path / "baselines.json")
    baseline = TaskBaseline(task_id="t")
    for name, (value, higher_is_better) in metrics.items():
        baseline.add_metric(
            name, value, std=0.0, sample_size=10, higher_is_better=higher_is_better
        )
    manager.set_baseline(baseline)
    return manager


def _compare(
    tmp_path: Path, current: list[float], *, higher_is_better: bool, baseline_value: float = 0.0
) -> dict[str, Any]:
    manager = _manager(tmp_path, {"m": (baseline_value, higher_is_better)})
    result: dict[str, Any] = manager.compare_to_baseline_with_ci(
        "t", {"m": current}, n_bootstrap=200
    )
    return result["m"]


@pytest.mark.parametrize(
    ("higher_is_better", "current", "is_regression"),
    [
        (True, UP, False),
        (True, DOWN, True),
        (False, UP, True),
        (False, DOWN, False),
    ],
    ids=["higher-up", "higher-down", "lower-up", "lower-down"],
)
def test_relative_delta_from_zero_is_unavailable_and_the_verdict_keeps_its_direction(
    tmp_path: Path, higher_is_better: bool, current: list[float], is_regression: bool
) -> None:
    row = _compare(tmp_path, current, higher_is_better=higher_is_better)

    assert row["relative_delta"] is None
    assert row["unavailable_fields"] == ["relative_delta"]
    assert row["delta"] == pytest.approx(sum(current) / len(current))
    assert row["is_significant"] is True
    assert row["is_regression"] is is_regression


def test_nonzero_baseline_row_keeps_its_shape(tmp_path: Path) -> None:
    row = _compare(tmp_path, UP, higher_is_better=True, baseline_value=4.0)

    assert set(row) == {
        "baseline", "current", "delta", "relative_delta", "ci_lower", "ci_upper",
        "confidence", "is_significant", "is_regression", "cohens_d",
        "effect_magnitude", "p_value", "higher_is_better",
    }
    assert row["relative_delta"] == pytest.approx(0.25)


def test_zero_baseline_comparison_round_trips_through_strict_json(tmp_path: Path) -> None:
    manager = _manager(tmp_path, {"pnl": (0.0, True), "loss": (0.0, False)})
    result = manager.compare_to_baseline_with_ci(
        "t", {"pnl": DOWN, "loss": UP}, n_bootstrap=200
    )

    encoded = json.dumps(result, allow_nan=False)

    assert json.loads(encoded) == result
