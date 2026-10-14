"""Deterministic canaries observe both directions without redefining regression."""

import json
from pathlib import Path

import pytest

from tracelens.baselines.manager import BaselineManager, MetricBaseline, TaskBaseline


def make_manager(
    tmp_path: Path, baseline_value: float = 120.0, absolute: float | None = 0.0,
    relative: float | None = None, higher_is_better: bool = True,
) -> BaselineManager:
    manager = BaselineManager(tmp_path / "baseline.json")
    baseline = TaskBaseline(task_id="replay")
    baseline.add_metric(
        "count", baseline_value, change_tolerance_absolute=absolute,
        change_tolerance_relative=relative, higher_is_better=higher_is_better,
    )
    manager.set_baseline(baseline)
    return manager


@pytest.mark.parametrize("higher_is_better", [True, False])
@pytest.mark.parametrize("current,changed", [(119.0, True), (120.0, False), (121.0, True)])
def test_exact_canary_flags_both_directions(
    tmp_path: Path, higher_is_better: bool, current: float, changed: bool,
) -> None:
    manager = make_manager(tmp_path, higher_is_better=higher_is_better)
    row = manager.compare_to_baseline("replay", {"count": current})["count"]
    assert row["changed"] is changed
    assert row["delta"] == current - 120.0
    assert row["regression"] is False  # No regression threshold was configured.


@pytest.mark.parametrize("current,changed", [(113.0, False), (127.0, False), (112.0, True), (128.0, True)])
def test_absolute_boundary_is_inclusive(tmp_path: Path, current: float, changed: bool) -> None:
    manager = make_manager(tmp_path, absolute=7.0)
    assert manager.compare_to_baseline("replay", {"count": current})["count"]["changed"] is changed


@pytest.mark.parametrize("baseline,current,changed", [
    (100.0, 105.0, False), (100.0, 95.0, False), (100.0, 106.0, True),
    (-100.0, -105.0, False), (-100.0, -95.0, False), (-100.0, -106.0, True),
    (0.0, 0.0, False), (0.0, 0.01, True), (0.0, -0.01, True),
])
def test_relative_tolerance_uses_baseline_magnitude(
    tmp_path: Path, baseline: float, current: float, changed: bool,
) -> None:
    manager = make_manager(tmp_path, baseline_value=baseline, absolute=None, relative=0.05)
    assert manager.compare_to_baseline("replay", {"count": current})["count"]["changed"] is changed


@pytest.mark.parametrize("absolute,relative", [(3.0, 0.1), (10.0, 0.02)])
def test_either_enabled_limit_can_trigger_change(
    tmp_path: Path, absolute: float, relative: float,
) -> None:
    manager = make_manager(tmp_path, baseline_value=100, absolute=absolute, relative=relative)
    assert manager.compare_to_baseline("replay", {"count": 105.0})["count"]["changed"] is True


def test_changed_does_not_replace_a_regression_verdict(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    baseline = manager.get_baseline("replay")
    assert baseline is not None
    baseline.metrics["count"].regression_threshold_absolute = -5
    up = manager.compare_to_baseline("replay", {"count": 140.0})["count"]
    down = manager.compare_to_baseline("replay", {"count": 110.0})["count"]
    assert up["changed"] and not up["regression"]
    assert down["changed"] and down["regression"]


def test_no_tolerance_preserves_the_legacy_row(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, absolute=None)
    assert "changed" not in manager.compare_to_baseline("replay", {"count": 140})["count"]


@pytest.mark.parametrize("field", ["change_tolerance_absolute", "change_tolerance_relative"])
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_tolerance_must_be_finite_and_nonnegative(field: str, invalid: float) -> None:
    with pytest.raises(ValueError):
        MetricBaseline(metric_name="count", baseline_value=120, **{field: invalid})


@pytest.mark.parametrize("baseline,current", [(120, float("nan")), (float("inf"), 120)])
def test_nonfinite_values_cannot_clear_a_tolerance_check(
    tmp_path: Path, baseline: float, current: float,
) -> None:
    manager = make_manager(tmp_path, baseline_value=baseline)
    with pytest.raises(ValueError, match="finite"):
        manager.compare_to_baseline("replay", {"count": current})


def test_an_invalid_assigned_tolerance_cannot_clear_the_gate(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)
    baseline = manager.get_baseline("replay")
    assert baseline is not None
    baseline.metrics["count"].change_tolerance_absolute = float("nan")
    with pytest.raises(ValueError, match="tolerance"):
        manager.compare_to_baseline("replay", {"count": 121})


@pytest.mark.parametrize("values,changed", [([119.0, 121.0], False), ([121.0, 121.0], True)])
def test_ci_checks_the_mean_separately_from_significance(
    tmp_path: Path, values: list[float], changed: bool,
) -> None:
    manager = make_manager(tmp_path)
    row = manager.compare_to_baseline_with_ci("replay", {"count": values}, n_bootstrap=100)["count"]
    assert row["changed"] is changed
    assert row["is_regression"] is False
    assert "ci_lower" in row


@pytest.mark.parametrize("operation", ["reload", "update", "force_promote", "try_promote"])
def test_write_paths_preserve_the_tolerances(tmp_path: Path, operation: str) -> None:
    manager = make_manager(tmp_path, absolute=0.0, relative=0.01)
    if operation == "reload":
        manager.save()
        manager = BaselineManager(tmp_path / "baseline.json")
    elif operation == "update":
        manager.update_baseline("replay", {"count": 140})
    elif operation == "force_promote":
        manager.force_promote("replay", {"count": 140})
    else:
        promoted, _ = manager.try_promote("replay", {"count": 140}, sample_size=20)
        assert promoted
    stored = manager.get_baseline("replay")
    assert stored is not None
    assert stored.metrics["count"].change_tolerance_absolute == 0.0
    assert stored.metrics["count"].change_tolerance_relative == 0.01
    if operation not in {"reload"}:
        old = stored.previous_versions[-1]["metrics"]["count"]
        assert old["change_tolerance_absolute"] == 0.0
        assert old["change_tolerance_relative"] == 0.01


def test_explicit_threshold_reset_clears_change_tolerances(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, absolute=0.0, relative=0.01)
    manager.update_baseline("replay", {"count": 140}, keep_thresholds=False)
    assert "changed" not in manager.compare_to_baseline("replay", {"count": 141})["count"]


def test_zero_tolerance_has_no_hidden_float_epsilon(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, baseline_value=0.3)
    assert manager.compare_to_baseline("replay", {"count": 0.1 + 0.2})["count"]["changed"] is True


def test_a_legacy_file_without_tolerance_fields_keeps_its_dictionary(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, absolute=None)
    manager.save()
    path = tmp_path / "baseline.json"
    data = json.loads(path.read_text())
    metric = data["replay"]["metrics"]["count"]
    metric.pop("change_tolerance_absolute")
    metric.pop("change_tolerance_relative")
    path.write_text(json.dumps(data))
    row = BaselineManager(path).compare_to_baseline("replay", {"count": 140})["count"]
    assert row == {
        "baseline": 120.0, "current": 140, "delta": 20.0,
        "relative_change": 20 / 120, "regression": False, "z_score": 0.0,
        "higher_is_better": True,
    }


@pytest.mark.parametrize("relative,changed", [(1.9, True), (2.0, False), (3.0, False)])
def test_overflow_cannot_turn_relative_movement_into_stability(
    tmp_path: Path, relative: float, changed: bool,
) -> None:
    manager = make_manager(tmp_path, baseline_value=1e308, absolute=None, relative=relative)
    assert manager.compare_to_baseline("replay", {"count": -1e308})["count"]["changed"] is changed


@pytest.mark.parametrize("field", ["change_tolerance_absolute", "change_tolerance_relative"])
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf")])
def test_invalid_assigned_tolerance_cannot_be_persisted_as_disabled(
    tmp_path: Path, field: str, invalid: float,
) -> None:
    manager = make_manager(tmp_path)
    manager.save()
    path = tmp_path / "baseline.json"
    frozen = path.read_bytes()
    baseline = manager.get_baseline("replay")
    assert baseline is not None
    setattr(baseline.metrics["count"], field, invalid)
    with pytest.raises(ValueError, match="tolerance"):
        manager.save()
    assert path.read_bytes() == frozen
    restored = BaselineManager(path).compare_to_baseline("replay", {"count": 121})
    assert restored["count"]["changed"] is True
