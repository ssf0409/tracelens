"""Tests for baseline promotion semantics."""

import tempfile
from pathlib import Path

import pytest

from eval_kit.baselines.manager import (
    BaselineManager,
    BaselineType,
    PromotionPolicy,
    TaskBaseline,
)


class TestBaselineType:
    """Tests for BaselineType enum."""

    def test_baseline_types_exist(self):
        """Test that all baseline types exist."""
        assert BaselineType.CANARY == "canary"
        assert BaselineType.CAPABILITY == "capability"
        assert BaselineType.EXPERIMENTAL == "experimental"


class TestPromotionPolicy:
    """Tests for PromotionPolicy."""

    def test_default_policy(self):
        """Test default promotion policy allows auto-promotion."""
        policy = PromotionPolicy()
        assert policy.allow_auto_promotion is True
        assert policy.min_improvement_relative == 0.05
        assert policy.min_samples == 10
        assert policy.required_confidence == 0.95

    def test_custom_policy(self):
        """Test custom promotion policy."""
        policy = PromotionPolicy(
            allow_auto_promotion=False,
            min_improvement_relative=0.10,
            min_samples=20,
            require_all_metrics_improve=True,
        )
        assert policy.allow_auto_promotion is False
        assert policy.min_improvement_relative == 0.10
        assert policy.min_samples == 20


class TestTaskBaselinePromotionProperties:
    """Tests for TaskBaseline promotion-related properties."""

    def test_is_canary(self):
        """Test is_canary property."""
        canary = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CANARY,
        )
        capability = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CAPABILITY,
        )

        assert canary.is_canary is True
        assert capability.is_canary is False

    def test_allows_auto_promotion(self):
        """Test allows_auto_promotion property."""
        canary = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CANARY,
        )
        capability = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CAPABILITY,
        )
        disabled = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CAPABILITY,
            promotion_policy=PromotionPolicy(allow_auto_promotion=False),
        )

        assert canary.allows_auto_promotion is False
        assert capability.allows_auto_promotion is True
        assert disabled.allows_auto_promotion is False


class TestTaskBaselineCanPromote:
    """Tests for TaskBaseline.can_promote method."""

    def test_canary_cannot_promote(self):
        """Test that canary baselines cannot auto-promote."""
        baseline = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CANARY,
        )
        baseline.add_metric("score", 0.8)

        can_promote, reason = baseline.can_promote({"score": 0.9}, sample_size=100)

        assert can_promote is False
        assert "Canary" in reason

    def test_insufficient_samples(self):
        """Test promotion fails with insufficient samples."""
        baseline = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CAPABILITY,
            promotion_policy=PromotionPolicy(min_samples=10),
        )
        baseline.add_metric("score", 0.8)

        can_promote, reason = baseline.can_promote({"score": 0.9}, sample_size=5)

        assert can_promote is False
        assert "samples" in reason.lower()

    def test_insufficient_improvement(self):
        """Test promotion fails without sufficient improvement."""
        baseline = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CAPABILITY,
            promotion_policy=PromotionPolicy(
                min_samples=1,
                min_improvement_relative=0.10,  # 10% required
            ),
        )
        baseline.add_metric("score", 0.8)

        # Only 1% improvement (0.808 vs 0.8)
        can_promote, reason = baseline.can_promote({"score": 0.808}, sample_size=10)

        assert can_promote is False

    def test_successful_promotion_criteria(self):
        """Test promotion succeeds when all criteria met."""
        baseline = TaskBaseline(
            task_id="test",
            baseline_type=BaselineType.CAPABILITY,
            promotion_policy=PromotionPolicy(
                min_samples=10,
                min_improvement_relative=0.05,  # 5% required
            ),
        )
        baseline.add_metric("score", 0.8)

        # 10% improvement
        can_promote, reason = baseline.can_promote({"score": 0.88}, sample_size=20)

        assert can_promote is True
        assert "criteria met" in reason.lower()


class TestTaskBaselinePromote:
    """Tests for TaskBaseline.promote method."""

    def test_promote_archives_previous(self):
        """Test that promotion archives previous version."""
        baseline = TaskBaseline(
            task_id="test",
            version="1.0.0",
        )
        baseline.add_metric("score", 0.8)

        baseline.promote({"score": 0.9}, sample_size=10)

        assert len(baseline.previous_versions) == 1
        assert baseline.previous_versions[0]["version"] == "1.0.0"

    def test_promote_increments_version(self):
        """Test that promotion increments version."""
        baseline = TaskBaseline(
            task_id="test",
            version="1.0.0",
        )
        baseline.add_metric("score", 0.8)

        baseline.promote({"score": 0.9}, sample_size=10)

        assert baseline.version == "1.0.1"

    def test_promote_updates_metrics(self):
        """Test that promotion updates metric values."""
        baseline = TaskBaseline(task_id="test")
        baseline.add_metric("score", 0.8, absolute_threshold=-0.1)

        baseline.promote({"score": 0.9}, sample_size=10)

        metric = baseline.get_metric("score")
        assert metric.baseline_value == 0.9
        # Thresholds are preserved
        assert metric.regression_threshold_absolute == -0.1

    def test_promote_updates_fingerprint(self):
        """Test that promotion can update fingerprint."""
        baseline = TaskBaseline(task_id="test")
        baseline.add_metric("score", 0.8)

        baseline.promote(
            {"score": 0.9},
            sample_size=10,
            fingerprint="newfingerprint123456",
        )

        assert baseline.fingerprint == "newfingerprint123456"
        assert baseline.fingerprint_short == "newfingerprint123456"[:12]

    def test_promote_increments_count(self):
        """Test that promotion increments promotion count."""
        baseline = TaskBaseline(task_id="test")
        baseline.add_metric("score", 0.8)

        baseline.promote({"score": 0.85}, sample_size=10)
        baseline.promote({"score": 0.90}, sample_size=10)

        assert baseline.promotion_count == 2


class TestBaselineManagerCanaryCreation:
    """Tests for BaselineManager canary baseline creation."""

    def test_create_canary_baseline(self):
        """Test creating a canary baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            baseline = manager.create_canary_baseline(
                task_id="safety_test",
                metrics={"pass_rate": 0.99},
                fingerprint="abc123def456789",
            )

            assert baseline.baseline_type == BaselineType.CANARY
            assert baseline.fingerprint == "abc123def456789"
            assert baseline.allows_auto_promotion is False

    def test_create_canary_requires_fingerprint(self):
        """Test that canary creation requires fingerprint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            with pytest.raises(ValueError, match="fingerprint"):
                manager.create_canary_baseline(
                    task_id="safety_test",
                    metrics={"pass_rate": 0.99},
                    fingerprint="",  # Empty fingerprint
                )


class TestBaselineManagerCapabilityCreation:
    """Tests for BaselineManager capability baseline creation."""

    def test_create_capability_baseline(self):
        """Test creating a capability baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            baseline = manager.create_capability_baseline(
                task_id="quality_test",
                metrics={"score": 0.85},
            )

            assert baseline.baseline_type == BaselineType.CAPABILITY
            assert baseline.allows_auto_promotion is True

    def test_create_capability_with_policy(self):
        """Test creating capability baseline with custom policy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            policy = PromotionPolicy(
                min_improvement_relative=0.10,
                min_samples=50,
            )

            baseline = manager.create_capability_baseline(
                task_id="quality_test",
                metrics={"score": 0.85},
                promotion_policy=policy,
            )

            assert baseline.promotion_policy.min_improvement_relative == 0.10
            assert baseline.promotion_policy.min_samples == 50


class TestBaselineManagerTryPromote:
    """Tests for BaselineManager.try_promote method."""

    def test_try_promote_no_baseline(self):
        """Test try_promote with no existing baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            promoted, reason = manager.try_promote(
                task_id="nonexistent",
                current_metrics={"score": 0.9},
            )

            assert promoted is False
            assert "No baseline" in reason

    def test_try_promote_canary_fails(self):
        """Test try_promote on canary baseline fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            manager.create_canary_baseline(
                task_id="safety",
                metrics={"pass_rate": 0.95},
                fingerprint="abc123",
            )

            promoted, reason = manager.try_promote(
                task_id="safety",
                current_metrics={"pass_rate": 0.99},
                sample_size=100,
            )

            assert promoted is False
            assert "Canary" in reason

    def test_try_promote_success(self):
        """Test successful try_promote."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            manager.create_capability_baseline(
                task_id="quality",
                metrics={"score": 0.8},
            )

            promoted, reason = manager.try_promote(
                task_id="quality",
                current_metrics={"score": 0.88},  # 10% improvement
                sample_size=20,
            )

            assert promoted is True

            # Check baseline was updated
            baseline = manager.get_baseline("quality")
            assert baseline.get_metric("score").baseline_value == 0.88


class TestBaselineManagerForcePromote:
    """Tests for BaselineManager.force_promote method."""

    def test_force_promote_canary(self):
        """Test force_promote works on canary baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            manager.create_canary_baseline(
                task_id="safety",
                metrics={"pass_rate": 0.95},
                fingerprint="abc123",
            )

            baseline = manager.force_promote(
                task_id="safety",
                current_metrics={"pass_rate": 0.99},
                reason="manual_override",
            )

            assert baseline.get_metric("pass_rate").baseline_value == 0.99
            assert "force:" in baseline.last_promotion_reason

    def test_force_promote_no_baseline(self):
        """Test force_promote raises on nonexistent baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            with pytest.raises(ValueError, match="No baseline"):
                manager.force_promote(
                    task_id="nonexistent",
                    current_metrics={"score": 0.9},
                )


class TestBaselineManagerListMethods:
    """Tests for BaselineManager list methods."""

    def test_list_canary_baselines(self):
        """Test listing canary baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            manager.create_canary_baseline(
                "safety1", {"score": 0.9}, "fp1"
            )
            manager.create_canary_baseline(
                "safety2", {"score": 0.9}, "fp2"
            )
            manager.create_capability_baseline(
                "quality", {"score": 0.8}
            )

            canaries = manager.list_canary_baselines()

            assert len(canaries) == 2
            assert "safety1" in canaries
            assert "safety2" in canaries
            assert "quality" not in canaries

    def test_list_capability_baselines(self):
        """Test listing capability baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = BaselineManager(Path(tmpdir) / "baselines.json")

            manager.create_canary_baseline(
                "safety", {"score": 0.9}, "fp1"
            )
            manager.create_capability_baseline(
                "quality1", {"score": 0.8}
            )
            manager.create_capability_baseline(
                "quality2", {"score": 0.8}
            )

            capabilities = manager.list_capability_baselines()

            assert len(capabilities) == 2
            assert "quality1" in capabilities
            assert "quality2" in capabilities
            assert "safety" not in capabilities


class TestBaselineManagerSaveLoad:
    """Tests for saving and loading with new fields."""

    def test_save_load_preserves_type(self):
        """Test that baseline type is preserved on save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baselines.json"

            # Create and save
            manager1 = BaselineManager(path)
            manager1.create_canary_baseline(
                "safety", {"score": 0.9}, "fingerprint123"
            )
            manager1.save()

            # Load in new manager
            manager2 = BaselineManager(path)
            baseline = manager2.get_baseline("safety")

            assert baseline.baseline_type == BaselineType.CANARY
            assert baseline.fingerprint == "fingerprint123"

    def test_save_load_preserves_policy(self):
        """Test that promotion policy is preserved on save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "baselines.json"

            # Create and save
            manager1 = BaselineManager(path)
            manager1.create_capability_baseline(
                "quality",
                {"score": 0.8},
                promotion_policy=PromotionPolicy(
                    min_improvement_relative=0.15,
                    min_samples=30,
                ),
            )
            manager1.save()

            # Load in new manager
            manager2 = BaselineManager(path)
            baseline = manager2.get_baseline("quality")

            assert baseline.promotion_policy.min_improvement_relative == 0.15
            assert baseline.promotion_policy.min_samples == 30
