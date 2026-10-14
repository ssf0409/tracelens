"""A disappearing measurement must not silently clear a strict canary."""

from pathlib import Path

import numpy as np
import pytest

from tracelens.baselines.manager import BaselineManager, TaskBaseline


@pytest.fixture
def manager(tmp_path: Path) -> BaselineManager:
    manager = BaselineManager(tmp_path / "baselines.json")
    baseline = TaskBaseline(task_id="replay")
    baseline.add_metric("decisions", 120, absolute_threshold=0)
    baseline.add_metric("settled", 10, absolute_threshold=0)
    manager.set_baseline(baseline)
    return manager


def test_strict_reports_both_missing_sides(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline(
        "replay", {"decisions": 120, "new_reason": 0}, coverage="strict"
    )
    assert result["decisions"]["coverage"] == "matched"
    assert result["settled"] == {
        "coverage": "missing_from_current", "baseline": 10.0,
        "current": None, "regression": None,
    }
    assert result["new_reason"] == {
        "coverage": "missing_from_baseline", "baseline": None,
        "current": 0, "regression": None,
    }


def test_strict_empty_current_reports_every_expected_metric(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline("replay", {}, coverage="strict")
    assert set(result) == {"decisions", "settled"}
    assert all(row["coverage"] == "missing_from_current" for row in result.values())


def test_default_dictionary_is_unchanged(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline("replay", {"decisions": 120, "new_reason": 0})
    assert result == {"decisions": {
        "baseline": 120.0, "current": 120, "delta": 0.0, "relative_change": 0.0,
        "regression": False, "z_score": 0.0, "higher_is_better": True,
    }}
    assert manager.compare_to_baseline("replay", {}) == {}


@pytest.mark.parametrize("coverage", ["intersection", "strict"])
def test_no_task_baseline_is_explicit(manager: BaselineManager, coverage: str) -> None:
    assert manager.compare_to_baseline("unknown", {}, coverage=coverage) == {"_no_baseline": True}


@pytest.mark.parametrize("method", ["compare_to_baseline", "compare_to_baseline_with_ci"])
def test_invalid_mode_cannot_silently_fall_back(manager: BaselineManager, method: str) -> None:
    with pytest.raises(ValueError, match="coverage"):
        getattr(manager, method)("unknown", {}, coverage="strcit")


def test_ci_reports_missing_metrics_without_inventing_estimates(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline_with_ci(
        "replay", {"new_reason": [1.0, 1.0]}, coverage="strict", n_bootstrap=100
    )
    assert result["settled"]["coverage"] == "missing_from_current"
    assert result["settled"]["is_regression"] is None
    assert result["new_reason"]["coverage"] == "missing_from_baseline"
    assert result["new_reason"]["current"] == [1.0, 1.0]
    assert result["new_reason"]["is_regression"] is None


def test_ci_empty_samples_are_unmeasured_in_strict_mode(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline_with_ci("replay", {"decisions": []}, coverage="strict")
    assert result["decisions"]["coverage"] == "missing_from_current"


def test_ci_matched_rows_keep_the_estimate(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline_with_ci(
        "replay", {"decisions": [120.0, 120.0]}, coverage="strict", n_bootstrap=100
    )
    assert result["decisions"]["coverage"] == "matched"
    assert result["decisions"]["current"]["mean"] == 120.0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_strict_rejects_nonfinite_measurements(manager: BaselineManager, value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        manager.compare_to_baseline("replay", {"decisions": value}, coverage="strict")


def test_ci_cannot_hide_a_nonfinite_sample_behind_a_valid_one(manager: BaselineManager) -> None:
    with pytest.raises(ValueError, match="finite"):
        manager.compare_to_baseline_with_ci(
            "replay", {"decisions": [float("nan"), 120.0]}, coverage="strict"
        )


def test_an_unknown_empty_sample_list_is_still_a_new_metric(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline_with_ci("replay", {"new_reason": []}, coverage="strict")
    assert result["new_reason"]["coverage"] == "missing_from_baseline"
    assert result["new_reason"]["current"] == []


def test_ci_numpy_samples_are_measured_by_length_not_truthiness(manager: BaselineManager) -> None:
    # A one-element array of 0.0 is falsy; a longer array has no truth value.
    result = manager.compare_to_baseline_with_ci(
        "replay", {"decisions": np.array([0.0]), "settled": np.array([10.0, 10.0])},
        coverage="strict", n_bootstrap=100,
    )
    assert result["decisions"]["coverage"] == "matched"
    assert result["decisions"]["current"]["mean"] == 0.0
    assert result["decisions"]["is_regression"] is True
    assert result["settled"]["coverage"] == "matched"


def test_ci_an_empty_numpy_sample_is_unmeasured(manager: BaselineManager) -> None:
    result = manager.compare_to_baseline_with_ci(
        "replay", {"decisions": np.array([]), "settled": np.array([10.0])},
        coverage="strict", n_bootstrap=100,
    )
    assert result["decisions"]["coverage"] == "missing_from_current"

