"""Tests for baseline manager."""

from pathlib import Path

import pytest

from tracelens.baselines.manager import (
    BaselineManager,
    MetricBaseline,
    TaskBaseline,
)
from tracelens.core.decision_spec import DecisionSpec, InfraConfig


class TestMetricBaseline:
    """Tests for MetricBaseline model."""

    def test_creation_minimal(self):
        """Test creating with minimal fields."""
        mb = MetricBaseline(
            metric_name="accuracy",
            baseline_value=0.95,
        )

        assert mb.metric_name == "accuracy"
        assert mb.baseline_value == 0.95
        assert mb.std_deviation == 0.0
        assert mb.higher_is_better is True

    def test_creation_full(self):
        """Test creating with all fields."""
        mb = MetricBaseline(
            metric_name="max_drawdown",
            baseline_value=-0.15,
            std_deviation=0.05,
            sample_size=100,
            regression_threshold_absolute=-0.05,
            regression_threshold_relative=0.20,
            higher_is_better=False,
        )

        assert mb.metric_name == "max_drawdown"
        assert mb.higher_is_better is False
        assert mb.regression_threshold_absolute == -0.05


class TestTaskBaseline:
    """Tests for TaskBaseline model."""

    def test_creation(self):
        """Test creating a task baseline."""
        baseline = TaskBaseline(
            task_id="test_task",
            task_name="Test Task",
        )

        assert baseline.task_id == "test_task"
        assert baseline.version == "1.0.0"
        assert baseline.metrics == {}

    def test_add_metric(self, sample_baseline: TaskBaseline):
        """Test adding metrics."""
        assert "sharpe_ratio" in sample_baseline.metrics
        assert sample_baseline.metrics["sharpe_ratio"].baseline_value == 1.2

    def test_get_metric(self, sample_baseline: TaskBaseline):
        """Test getting a metric."""
        metric = sample_baseline.get_metric("sharpe_ratio")
        assert metric is not None
        assert metric.baseline_value == 1.2

        metric = sample_baseline.get_metric("nonexistent")
        assert metric is None


class TestBaselineManager:
    """Tests for BaselineManager class."""

    def test_creation_empty(self, temp_baselines_file: Path):
        """Test creating manager with no existing file."""
        manager = BaselineManager(temp_baselines_file)

        assert manager.list_tasks() == []

    def test_save_and_load(self, temp_baselines_file: Path, sample_baseline: TaskBaseline):
        """Test saving and loading baselines."""
        manager = BaselineManager(temp_baselines_file)
        manager.set_baseline(sample_baseline)
        manager.save()

        # Create new manager and verify loaded
        manager2 = BaselineManager(temp_baselines_file)
        loaded = manager2.get_baseline("btc_backtest")

        assert loaded is not None
        assert loaded.task_id == "btc_backtest"
        assert "sharpe_ratio" in loaded.metrics

    def test_update_baseline(self, temp_baselines_file: Path):
        """Test updating a baseline."""
        manager = BaselineManager(temp_baselines_file)

        # Create initial baseline
        baseline = manager.update_baseline(
            task_id="test_task",
            metrics={"accuracy": 0.9},
        )

        assert baseline.version == "1.0.0"
        assert baseline.get_metric("accuracy").baseline_value == 0.9

        # Update baseline
        updated = manager.update_baseline(
            task_id="test_task",
            metrics={"accuracy": 0.95},
        )

        assert updated.version == "1.0.1"
        assert updated.get_metric("accuracy").baseline_value == 0.95
        assert len(updated.previous_versions) == 1

    def test_compare_to_baseline_no_baseline(self, temp_baselines_file: Path):
        """Test comparison when no baseline exists."""
        manager = BaselineManager(temp_baselines_file)

        result = manager.compare_to_baseline(
            "nonexistent",
            {"accuracy": 0.9},
        )

        assert result == {"_no_baseline": True}

    def test_compare_to_baseline_no_regression(
        self, temp_baselines_file: Path, sample_baseline: TaskBaseline
    ):
        """Test comparison with no regression."""
        manager = BaselineManager(temp_baselines_file)
        manager.set_baseline(sample_baseline)

        # Current values same or better than baseline
        result = manager.compare_to_baseline(
            "btc_backtest",
            {"sharpe_ratio": 1.3, "max_drawdown": -0.12, "win_rate": 0.58},
        )

        assert "sharpe_ratio" in result
        assert result["sharpe_ratio"]["regression"] is False
        assert result["sharpe_ratio"]["delta"] == pytest.approx(0.1, rel=0.01)

    def test_compare_to_baseline_with_regression(
        self, temp_baselines_file: Path, sample_baseline: TaskBaseline
    ):
        """Test comparison detecting regression."""
        manager = BaselineManager(temp_baselines_file)
        manager.set_baseline(sample_baseline)

        # Sharpe drops below threshold
        result = manager.compare_to_baseline(
            "btc_backtest",
            {"sharpe_ratio": 0.9},  # Drop of 0.3 (> threshold of 0.2)
        )

        assert result["sharpe_ratio"]["regression"] is True
        assert result["sharpe_ratio"]["delta"] == pytest.approx(-0.3, rel=0.01)

    def test_compare_to_baseline_lower_is_better(
        self, temp_baselines_file: Path, sample_baseline: TaskBaseline
    ):
        """Test comparison for metric where lower is better."""
        manager = BaselineManager(temp_baselines_file)
        manager.set_baseline(sample_baseline)

        # Max drawdown gets worse (more negative)
        result = manager.compare_to_baseline(
            "btc_backtest",
            {"max_drawdown": -0.25},  # Worse than baseline of -0.15
        )

        assert result["max_drawdown"]["regression"] is True

    def test_list_tasks(self, temp_baselines_file: Path, sample_baseline: TaskBaseline):
        """Test listing task IDs."""
        manager = BaselineManager(temp_baselines_file)
        manager.set_baseline(sample_baseline)

        tasks = manager.list_tasks()
        assert "btc_backtest" in tasks


class TestBaselineDecisionSpecRoundTrip:
    """TaskBaseline must carry the DecisionSpec that produced it, so the
    CLI gate can hand both specs to compare_with_specs() and apply the
    infra-noise band. A fingerprint string alone can detect a mismatch
    but cannot diff the infra sections."""

    def test_decision_spec_survives_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "baselines.json"
        manager = BaselineManager(path)
        baseline = TaskBaseline(
            task_id="t1",
            decision_spec=DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=2048)),
        )
        baseline.add_metric(
            metric_name="mean_score", value=0.2, std=0.001, sample_size=10
        )
        manager.set_baseline(baseline)
        manager.save()

        reloaded = BaselineManager(path).get_baseline("t1")
        assert reloaded is not None
        assert reloaded.decision_spec is not None
        assert reloaded.decision_spec.infra is not None
        assert reloaded.decision_spec.infra.memory_hard_limit_mb == 2048

    def test_decision_spec_defaults_to_none(self) -> None:
        baseline = TaskBaseline(task_id="t1")
        assert baseline.decision_spec is None


class TestDecisionSpecWritePath:
    """Every baseline write API must be able to record the DecisionSpec —
    otherwise the noise-aware feature only activates via hand-edited
    JSON, and promotion leaves a stale spec that fakes a permanent
    infra mismatch."""

    def _spec(self, memory_mb: int) -> "DecisionSpec":
        return DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=memory_mb))

    def test_update_baseline_stores_decision_spec(self, tmp_path: Path) -> None:
        manager = BaselineManager(tmp_path / "baselines.json")

        baseline = manager.update_baseline(
            "t1", {"pass_rate": 1.0}, decision_spec=self._spec(2048)
        )

        assert baseline.decision_spec is not None
        assert baseline.fingerprint == self._spec(2048).fingerprint

    def test_create_capability_baseline_stores_spec_and_derives_fingerprint(
        self, tmp_path: Path
    ) -> None:
        manager = BaselineManager(tmp_path / "baselines.json")
        spec = self._spec(2048)

        baseline = manager.create_capability_baseline(
            "t1", {"pass_rate": 1.0}, decision_spec=spec
        )

        assert baseline.decision_spec == spec
        assert baseline.fingerprint == spec.fingerprint

    def test_create_canary_baseline_accepts_spec_instead_of_fingerprint(
        self, tmp_path: Path
    ) -> None:
        manager = BaselineManager(tmp_path / "baselines.json")
        spec = self._spec(2048)

        baseline = manager.create_canary_baseline(
            "t1", {"pass_rate": 1.0}, decision_spec=spec
        )

        assert baseline.fingerprint == spec.fingerprint
        assert baseline.decision_spec == spec

    def test_promote_refreshes_spec_and_archives_the_old_one(
        self, tmp_path: Path
    ) -> None:
        manager = BaselineManager(tmp_path / "baselines.json")
        old_spec, new_spec = self._spec(2048), self._spec(512)
        baseline = manager.create_capability_baseline(
            "t1", {"pass_rate": 1.0}, sample_size=20, decision_spec=old_spec
        )

        baseline.promote(
            {"pass_rate": 1.0}, sample_size=20, decision_spec=new_spec
        )

        assert baseline.decision_spec == new_spec
        assert baseline.fingerprint == new_spec.fingerprint
        archived = baseline.previous_versions[-1]
        assert archived["fingerprint"] == old_spec.fingerprint
        assert archived["decision_spec"] is not None

    def test_force_promote_threads_decision_spec(self, tmp_path: Path) -> None:
        manager = BaselineManager(tmp_path / "baselines.json")
        manager.create_capability_baseline(
            "t1", {"pass_rate": 1.0}, decision_spec=self._spec(2048)
        )

        promoted = manager.force_promote(
            "t1", {"pass_rate": 1.0}, decision_spec=self._spec(512)
        )

        assert promoted.decision_spec == self._spec(512)
        assert promoted.fingerprint == self._spec(512).fingerprint
