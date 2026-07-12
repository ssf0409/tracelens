"""Baseline storage and retrieval.

Baselines store historical performance metrics that are used to
detect regressions when new evaluations are run.

Supports two baseline types:
- CANARY: Never auto-updates; represents golden performance floor
- CAPABILITY: Can auto-update on improvements; tracks current capability
"""

import json
import subprocess
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from tracelens.core._time import utc_now
from tracelens.core.decision_spec import DecisionSpec
from tracelens.statistics.inference import (
    compare_to_baseline_summary,
    estimate_metric,
)


class BaselineType(str, Enum):
    """Type of baseline determining update semantics.

    CANARY: Protected baseline that never auto-updates.
        - Represents the absolute performance floor
        - Only manually updated after explicit approval
        - Tied to a specific DecisionSpec fingerprint
        - Use for safety-critical or business-critical metrics

    CAPABILITY: Baseline that can auto-update on improvements.
        - Tracks the agent's current capability ceiling
        - Auto-updates when performance improves significantly
        - Maintains version history for rollback
        - Use for tracking progress and catching regressions

    EXPERIMENTAL: Baseline for experimental features.
        - Loose thresholds, high variance expected
        - Auto-updates more aggressively
        - Use during active development
    """

    CANARY = "canary"
    CAPABILITY = "capability"
    EXPERIMENTAL = "experimental"


class PromotionPolicy(BaseModel):
    """Policy for automatic baseline promotion.

    Controls when and how baselines can be automatically updated.
    """

    # Whether auto-promotion is allowed at all
    allow_auto_promotion: bool = True

    # Minimum improvement required for auto-promotion (relative, e.g., 0.05 = 5%)
    min_improvement_relative: float = 0.05

    # Minimum number of samples required for promotion
    min_samples: int = 10

    # Required confidence level (e.g., 0.95 for 95% CI)
    required_confidence: float = 0.95

    # Maximum staleness before baseline needs refresh (days)
    max_age_days: int | None = None

    # Require all metrics to improve (not just average)
    require_all_metrics_improve: bool = False

    # Require manual approval before promotion
    require_manual_approval: bool = False

    # Cooldown period between promotions (days)
    promotion_cooldown_days: int = 0


class MetricBaseline(BaseModel):
    """Baseline for a single metric.

    Stores the expected value, standard deviation, and thresholds
    for regression detection.
    """

    metric_name: str
    baseline_value: float
    std_deviation: float = 0.0
    sample_size: int = 1
    computed_at: datetime = Field(default_factory=utc_now)

    # Thresholds for regression detection
    # absolute: fail if (current - baseline) < threshold (e.g., -0.2)
    # relative: fail if (current - baseline) / baseline < threshold (e.g., -0.1 for 10% drop)
    regression_threshold_absolute: float | None = None
    regression_threshold_relative: float | None = None

    # Direction indicator for proper comparison
    higher_is_better: bool = True  # False for metrics like drawdown


class TaskBaseline(BaseModel):
    """Baseline for a complete task.

    Groups metric baselines together with metadata about when
    the baseline was created.

    Supports two types of baselines:
    - CANARY: Protected, never auto-updates (safety floor)
    - CAPABILITY: Can auto-update on improvements

    Example:
        # Create a canary baseline (protected)
        baseline = TaskBaseline(
            task_id="safety_check",
            baseline_type=BaselineType.CANARY,
            fingerprint="abc123def456",  # Tied to specific config
        )

        # Create a capability baseline (can auto-update)
        baseline = TaskBaseline(
            task_id="quality_score",
            baseline_type=BaselineType.CAPABILITY,
            promotion_policy=PromotionPolicy(min_improvement_relative=0.05),
        )
    """

    task_id: str
    task_name: str | None = None
    version: str = "1.0.0"
    environment: str = "development"

    # Baseline type and promotion policy
    baseline_type: BaselineType = BaselineType.CAPABILITY
    promotion_policy: PromotionPolicy = Field(default_factory=PromotionPolicy)

    # DecisionSpec fingerprint for reproducibility
    # For CANARY baselines, this is required and enforced
    fingerprint: str | None = None
    fingerprint_short: str | None = None

    # Full DecisionSpec captured when the baseline was recorded. A
    # fingerprint alone can detect a config mismatch but cannot diff the
    # infra sections — storing the spec is what lets
    # RegressionDetector.compare_with_specs() apply the infra-noise band.
    decision_spec: DecisionSpec | None = None

    # Metric baselines
    metrics: dict[str, MetricBaseline] = Field(default_factory=dict)

    # Metadata
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_promoted_at: datetime | None = None
    created_by: str = "manual"  # 'manual', 'ci', 'scheduled'
    git_commit: str | None = None
    git_branch: str | None = None

    # History for tracking changes (especially important for CAPABILITY baselines)
    previous_versions: list[dict[str, Any]] = Field(default_factory=list)

    # Promotion tracking
    promotion_count: int = 0
    last_promotion_reason: str | None = None

    def add_metric(
        self,
        metric_name: str,
        value: float,
        std: float = 0.0,
        sample_size: int = 1,
        absolute_threshold: float | None = None,
        relative_threshold: float | None = None,
        higher_is_better: bool = True,
    ) -> None:
        """Add or update a metric baseline."""
        self.metrics[metric_name] = MetricBaseline(
            metric_name=metric_name,
            baseline_value=value,
            std_deviation=std,
            sample_size=sample_size,
            regression_threshold_absolute=absolute_threshold,
            regression_threshold_relative=relative_threshold,
            higher_is_better=higher_is_better,
        )
        self.updated_at = utc_now()

    def get_metric(self, metric_name: str) -> MetricBaseline | None:
        """Get a specific metric baseline."""
        return self.metrics.get(metric_name)

    @property
    def is_canary(self) -> bool:
        """Check if this is a canary (protected) baseline."""
        return self.baseline_type == BaselineType.CANARY

    @property
    def allows_auto_promotion(self) -> bool:
        """Check if this baseline allows automatic promotion."""
        if self.baseline_type == BaselineType.CANARY:
            return False
        return self.promotion_policy.allow_auto_promotion

    @property
    def is_stale(self) -> bool:
        """Check if baseline is stale based on max_age_days policy."""
        if self.promotion_policy.max_age_days is None:
            return False
        age = (utc_now() - self.updated_at).days
        return age > self.promotion_policy.max_age_days

    @property
    def in_cooldown(self) -> bool:
        """Check if baseline is in promotion cooldown period."""
        if self.last_promoted_at is None:
            return False
        if self.promotion_policy.promotion_cooldown_days == 0:
            return False
        days_since = (utc_now() - self.last_promoted_at).days
        return days_since < self.promotion_policy.promotion_cooldown_days

    def can_promote(
        self,
        current_metrics: dict[str, float],
        sample_size: int = 1,
    ) -> tuple[bool, str]:
        """Check if baseline can be promoted with new metrics.

        Args:
            current_metrics: New metric values
            sample_size: Number of samples used to compute metrics

        Returns:
            Tuple of (can_promote, reason)
        """
        # Canary baselines never auto-promote
        if self.baseline_type == BaselineType.CANARY:
            return False, "Canary baselines require manual promotion"

        # Check if auto-promotion is allowed
        if not self.promotion_policy.allow_auto_promotion:
            return False, "Auto-promotion disabled for this baseline"

        # Check cooldown
        if self.in_cooldown:
            return False, "In promotion cooldown period"

        # Check sample size
        if sample_size < self.promotion_policy.min_samples:
            return (
                False,
                f"Insufficient samples: {sample_size} < {self.promotion_policy.min_samples}",
            )

        # Check for improvement
        improvements = []
        for metric_name, current_value in current_metrics.items():
            metric_baseline = self.get_metric(metric_name)
            if not metric_baseline:
                continue

            baseline_value = metric_baseline.baseline_value
            if baseline_value == 0:
                continue

            relative_change = (current_value - baseline_value) / abs(baseline_value)

            # Adjust for direction
            if not metric_baseline.higher_is_better:
                relative_change = -relative_change

            is_improvement = relative_change >= self.promotion_policy.min_improvement_relative
            improvements.append((metric_name, relative_change, is_improvement))

        if not improvements:
            return False, "No comparable metrics"

        # Check improvement requirements
        if self.promotion_policy.require_all_metrics_improve:
            if not all(imp for _, _, imp in improvements):
                failed = [name for name, _, imp in improvements if not imp]
                return False, f"Not all metrics improved: {failed}"
        else:
            # At least average improvement should be positive
            avg_improvement = sum(change for _, change, _ in improvements) / len(improvements)
            if avg_improvement < self.promotion_policy.min_improvement_relative:
                return False, f"Average improvement {avg_improvement:.2%} below threshold"

        # Check manual approval requirement
        if self.promotion_policy.require_manual_approval:
            return False, "Manual approval required for promotion"

        return True, "All promotion criteria met"

    def promote(
        self,
        current_metrics: dict[str, float],
        metric_stds: dict[str, float] | None = None,
        sample_size: int = 1,
        reason: str = "auto",
        fingerprint: str | None = None,
        decision_spec: DecisionSpec | None = None,
    ) -> None:
        """Promote baseline to new values.

        Archives current version and updates metrics.

        Args:
            current_metrics: New metric values
            metric_stds: Optional standard deviations
            sample_size: Number of samples
            reason: Reason for promotion
            fingerprint: Optional new fingerprint
            decision_spec: DecisionSpec of the run being promoted. Pass it
                whenever the baseline carries one — a fingerprint refresh
                without the matching spec leaves noise-aware comparison
                diffing against the pre-promotion infra config.
        """
        # Archive current version
        self.previous_versions.append({
            "version": self.version,
            "updated_at": self.updated_at.isoformat(),
            "metrics": {k: v.model_dump() for k, v in self.metrics.items()},
            "fingerprint": self.fingerprint,
            "decision_spec": (
                self.decision_spec.model_dump(mode="json")
                if self.decision_spec
                else None
            ),
        })

        # Update version
        parts = self.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        self.version = ".".join(parts)

        # Update metrics
        metric_stds = metric_stds or {}
        for metric_name, value in current_metrics.items():
            old_metric = self.get_metric(metric_name)

            # Preserve thresholds and direction from existing metric
            abs_threshold = None
            rel_threshold = None
            higher_is_better = True

            if old_metric:
                abs_threshold = old_metric.regression_threshold_absolute
                rel_threshold = old_metric.regression_threshold_relative
                higher_is_better = old_metric.higher_is_better

            self.add_metric(
                metric_name=metric_name,
                value=value,
                std=metric_stds.get(metric_name, 0.0),
                sample_size=sample_size,
                absolute_threshold=abs_threshold,
                relative_threshold=rel_threshold,
                higher_is_better=higher_is_better,
            )

        # Update metadata
        now = utc_now()
        self.updated_at = now
        self.last_promoted_at = now
        self.promotion_count += 1
        self.last_promotion_reason = reason

        if decision_spec is not None:
            self.decision_spec = decision_spec
            if fingerprint is None:
                fingerprint = decision_spec.fingerprint
        if fingerprint:
            self.fingerprint = fingerprint
            self.fingerprint_short = fingerprint[:12] if fingerprint else None


class BaselineManager:
    """Manages baseline storage and retrieval.

    Baselines are stored in a JSON file and can be versioned
    with git for tracking changes over time.

    Example:
        manager = BaselineManager("baselines/baselines.json")

        # Get existing baseline
        baseline = manager.get_baseline("btc_backtest")

        # Update baseline
        manager.update_baseline("btc_backtest", {"sharpe_ratio": 1.5})

        # Save changes
        manager.save()
    """

    def __init__(self, baselines_path: str | Path):
        """Initialize the baseline manager.

        Args:
            baselines_path: Path to the baselines JSON file
        """
        self.baselines_path = Path(baselines_path)
        self._baselines: dict[str, TaskBaseline] = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load baselines from JSON file."""
        if self.baselines_path.exists():
            with open(self.baselines_path) as f:
                data = json.load(f)

            for task_id, baseline_data in data.items():
                self._baselines[task_id] = self._parse_baseline(baseline_data)

    def _parse_baseline(self, data: dict[str, Any]) -> TaskBaseline:
        """Parse baseline data from JSON."""
        # Convert metrics dict
        metrics = {}
        for metric_name, metric_data in data.get("metrics", {}).items():
            metrics[metric_name] = MetricBaseline(**metric_data)

        # Parse promotion policy if present
        promotion_policy = PromotionPolicy()
        if "promotion_policy" in data:
            promotion_policy = PromotionPolicy(**data["promotion_policy"])

        # Parse baseline type
        baseline_type = BaselineType.CAPABILITY
        if "baseline_type" in data:
            baseline_type = BaselineType(data["baseline_type"])

        return TaskBaseline(
            task_id=data["task_id"],
            task_name=data.get("task_name"),
            version=data.get("version", "1.0.0"),
            environment=data.get("environment", "development"),
            baseline_type=baseline_type,
            promotion_policy=promotion_policy,
            fingerprint=data.get("fingerprint"),
            fingerprint_short=data.get("fingerprint_short"),
            decision_spec=(
                DecisionSpec.model_validate(data["decision_spec"])
                if data.get("decision_spec")
                else None
            ),
            metrics=metrics,
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data else utc_now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data else utc_now(),
            last_promoted_at=datetime.fromisoformat(data["last_promoted_at"])
            if "last_promoted_at" in data and data["last_promoted_at"] else None,
            created_by=data.get("created_by", "manual"),
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
            previous_versions=data.get("previous_versions", []),
            promotion_count=data.get("promotion_count", 0),
            last_promotion_reason=data.get("last_promotion_reason"),
        )

    def save(self) -> None:
        """Save baselines to JSON file."""
        self.baselines_path.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for task_id, baseline in self._baselines.items():
            data[task_id] = json.loads(baseline.model_dump_json())

        with open(self.baselines_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_baseline(self, task_id: str) -> TaskBaseline | None:
        """Get baseline for a task.

        Args:
            task_id: The task identifier

        Returns:
            TaskBaseline if found, None otherwise
        """
        return self._baselines.get(task_id)

    def set_baseline(self, baseline: TaskBaseline) -> None:
        """Set baseline for a task.

        Args:
            baseline: The task baseline to store
        """
        self._baselines[baseline.task_id] = baseline

    def update_baseline(
        self,
        task_id: str,
        metrics: dict[str, float],
        metric_stds: dict[str, float] | None = None,
        sample_size: int = 1,
        keep_thresholds: bool = True,
        decision_spec: DecisionSpec | None = None,
    ) -> TaskBaseline:
        """Update or create a baseline with new metric values.

        Args:
            task_id: The task identifier
            metrics: Dict of metric_name -> value
            metric_stds: Optional dict of metric_name -> std deviation
            sample_size: Number of samples used to compute metrics
            keep_thresholds: Keep existing thresholds when updating
            decision_spec: DecisionSpec of the run these metrics came
                from; stored on the baseline to enable noise-aware
                comparison

        Returns:
            The updated TaskBaseline
        """
        existing = self.get_baseline(task_id)

        if existing:
            # Archive current version
            existing.previous_versions.append({
                "version": existing.version,
                "updated_at": existing.updated_at.isoformat(),
                "metrics": {k: v.model_dump() for k, v in existing.metrics.items()},
            })

            # Update version
            parts = existing.version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            existing.version = ".".join(parts)

        else:
            existing = TaskBaseline(task_id=task_id)

        # Update metrics
        metric_stds = metric_stds or {}

        for metric_name, value in metrics.items():
            old_metric = existing.get_metric(metric_name)

            # Preserve thresholds if requested
            abs_threshold = None
            rel_threshold = None
            higher_is_better = True

            if keep_thresholds and old_metric:
                abs_threshold = old_metric.regression_threshold_absolute
                rel_threshold = old_metric.regression_threshold_relative
                higher_is_better = old_metric.higher_is_better

            existing.add_metric(
                metric_name=metric_name,
                value=value,
                std=metric_stds.get(metric_name, 0.0),
                sample_size=sample_size,
                absolute_threshold=abs_threshold,
                relative_threshold=rel_threshold,
                higher_is_better=higher_is_better,
            )

        if decision_spec is not None:
            existing.decision_spec = decision_spec
            if existing.fingerprint is None:
                existing.fingerprint = decision_spec.fingerprint
                existing.fingerprint_short = decision_spec.fingerprint_short

        existing.updated_at = utc_now()
        existing.git_commit = self._get_git_commit()

        self._baselines[task_id] = existing
        return existing

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.baselines_path.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            pass
        return None

    def list_tasks(self) -> list[str]:
        """List all task IDs with baselines."""
        return list(self._baselines.keys())

    def compare_to_baseline(
        self,
        task_id: str,
        current_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Compare current metrics to baseline.

        Args:
            task_id: The task identifier
            current_metrics: Dict of metric_name -> current value

        Returns:
            Dict of metric_name -> comparison dict with:
            - baseline: baseline value
            - current: current value
            - delta: absolute difference
            - relative_change: relative difference
            - regression: True if regression detected
            - z_score: standard score if std available
        """
        baseline = self.get_baseline(task_id)

        if not baseline:
            return {"_no_baseline": True}

        comparisons = {}

        for metric_name, current_value in current_metrics.items():
            metric_baseline = baseline.get_metric(metric_name)

            if not metric_baseline:
                continue

            baseline_value = metric_baseline.baseline_value
            delta = current_value - baseline_value

            # Calculate relative change (handle zero baseline)
            if baseline_value != 0:
                relative_change = delta / abs(baseline_value)
            else:
                relative_change = float("inf") if delta != 0 else 0.0

            # Check for regression
            regression = False

            # For metrics where higher is better
            if metric_baseline.higher_is_better:
                # Check absolute threshold (negative means decline)
                if (metric_baseline.regression_threshold_absolute is not None and
                        delta < metric_baseline.regression_threshold_absolute):
                    regression = True

                # Check relative threshold (negative means decline)
                if (metric_baseline.regression_threshold_relative is not None and
                        relative_change < -metric_baseline.regression_threshold_relative):
                    regression = True
            else:
                # For metrics where lower is better (e.g., drawdown)
                if (metric_baseline.regression_threshold_absolute is not None and
                        delta > -metric_baseline.regression_threshold_absolute):
                    regression = True

                if (metric_baseline.regression_threshold_relative is not None and
                        relative_change > metric_baseline.regression_threshold_relative):
                    regression = True

            # Calculate z-score if std available
            z_score = 0.0
            if metric_baseline.std_deviation > 0:
                z_score = delta / metric_baseline.std_deviation

            comparisons[metric_name] = {
                "baseline": baseline_value,
                "current": current_value,
                "delta": delta,
                "relative_change": relative_change,
                "regression": regression,
                "z_score": z_score,
                "higher_is_better": metric_baseline.higher_is_better,
            }

        return comparisons

    def create_canary_baseline(
        self,
        task_id: str,
        metrics: dict[str, float],
        fingerprint: str | None = None,
        metric_stds: dict[str, float] | None = None,
        sample_size: int = 1,
        task_name: str | None = None,
        decision_spec: DecisionSpec | None = None,
    ) -> TaskBaseline:
        """Create a protected canary baseline.

        Canary baselines never auto-update and are tied to a specific
        DecisionSpec fingerprint. Use for safety-critical metrics.

        Args:
            task_id: The task identifier
            metrics: Dict of metric_name -> value
            fingerprint: DecisionSpec fingerprint (required for canary;
                derived from decision_spec when omitted)
            metric_stds: Optional dict of metric_name -> std deviation
            sample_size: Number of samples used to compute metrics
            task_name: Optional human-readable name
            decision_spec: Full DecisionSpec of the recording run; stored
                for noise-aware comparison and used to derive the
                fingerprint when one isn't passed

        Returns:
            The created canary baseline

        Raises:
            ValueError: If neither fingerprint nor decision_spec is provided
        """
        if not fingerprint and decision_spec is not None:
            fingerprint = decision_spec.fingerprint
        if not fingerprint:
            raise ValueError(
                "Canary baselines require a fingerprint or decision_spec"
            )

        baseline = TaskBaseline(
            task_id=task_id,
            task_name=task_name,
            baseline_type=BaselineType.CANARY,
            fingerprint=fingerprint,
            fingerprint_short=fingerprint[:12],
            decision_spec=decision_spec,
            promotion_policy=PromotionPolicy(allow_auto_promotion=False),
            git_commit=self._get_git_commit(),
        )

        metric_stds = metric_stds or {}
        for metric_name, value in metrics.items():
            baseline.add_metric(
                metric_name=metric_name,
                value=value,
                std=metric_stds.get(metric_name, 0.0),
                sample_size=sample_size,
            )

        self._baselines[task_id] = baseline
        return baseline

    def create_capability_baseline(
        self,
        task_id: str,
        metrics: dict[str, float],
        metric_stds: dict[str, float] | None = None,
        sample_size: int = 1,
        task_name: str | None = None,
        promotion_policy: PromotionPolicy | None = None,
        fingerprint: str | None = None,
        decision_spec: DecisionSpec | None = None,
    ) -> TaskBaseline:
        """Create a capability baseline that can auto-update.

        Capability baselines track current agent capability and can
        be promoted when performance improves.

        Args:
            task_id: The task identifier
            metrics: Dict of metric_name -> value
            metric_stds: Optional dict of metric_name -> std deviation
            sample_size: Number of samples used to compute metrics
            task_name: Optional human-readable name
            promotion_policy: Custom promotion policy (default allows auto-promotion)
            fingerprint: Optional DecisionSpec fingerprint (derived from
                decision_spec when omitted)
            decision_spec: Full DecisionSpec of the recording run; stored
                for noise-aware comparison

        Returns:
            The created capability baseline
        """
        if fingerprint is None and decision_spec is not None:
            fingerprint = decision_spec.fingerprint

        baseline = TaskBaseline(
            task_id=task_id,
            task_name=task_name,
            baseline_type=BaselineType.CAPABILITY,
            fingerprint=fingerprint,
            fingerprint_short=fingerprint[:12] if fingerprint else None,
            decision_spec=decision_spec,
            promotion_policy=promotion_policy or PromotionPolicy(),
            git_commit=self._get_git_commit(),
        )

        metric_stds = metric_stds or {}
        for metric_name, value in metrics.items():
            baseline.add_metric(
                metric_name=metric_name,
                value=value,
                std=metric_stds.get(metric_name, 0.0),
                sample_size=sample_size,
            )

        self._baselines[task_id] = baseline
        return baseline

    def try_promote(
        self,
        task_id: str,
        current_metrics: dict[str, float],
        metric_stds: dict[str, float] | None = None,
        sample_size: int = 1,
        fingerprint: str | None = None,
        decision_spec: DecisionSpec | None = None,
    ) -> tuple[bool, str]:
        """Try to promote a baseline if criteria are met.

        This method checks if the baseline can be promoted and
        performs the promotion if allowed.

        Args:
            task_id: The task identifier
            current_metrics: New metric values
            metric_stds: Optional standard deviations
            sample_size: Number of samples
            fingerprint: New fingerprint for the promoted baseline
            decision_spec: DecisionSpec of the promoted run; stored on
                the baseline and used to derive the fingerprint when one
                isn't passed (see TaskBaseline.promote)

        Returns:
            Tuple of (was_promoted, reason)
        """
        baseline = self.get_baseline(task_id)

        if not baseline:
            return False, "No baseline exists"

        can_promote, reason = baseline.can_promote(current_metrics, sample_size)

        if not can_promote:
            return False, reason

        # Perform the promotion
        baseline.promote(
            current_metrics=current_metrics,
            metric_stds=metric_stds,
            sample_size=sample_size,
            reason="auto_promotion",
            fingerprint=fingerprint,
            decision_spec=decision_spec,
        )
        baseline.git_commit = self._get_git_commit()

        return True, "Baseline promoted successfully"

    def force_promote(
        self,
        task_id: str,
        current_metrics: dict[str, float],
        metric_stds: dict[str, float] | None = None,
        sample_size: int = 1,
        reason: str = "manual",
        fingerprint: str | None = None,
        decision_spec: DecisionSpec | None = None,
    ) -> TaskBaseline:
        """Force promote a baseline, bypassing policy checks.

        Use this for manual promotions or emergency updates.
        Even canary baselines can be force-promoted.

        Args:
            task_id: The task identifier
            current_metrics: New metric values
            metric_stds: Optional standard deviations
            sample_size: Number of samples
            reason: Reason for the forced promotion
            fingerprint: New fingerprint for the promoted baseline
            decision_spec: DecisionSpec of the promoted run; stored on
                the baseline and used to derive the fingerprint when one
                isn't passed (see TaskBaseline.promote)

        Returns:
            The promoted baseline

        Raises:
            ValueError: If no baseline exists
        """
        baseline = self.get_baseline(task_id)

        if not baseline:
            raise ValueError(f"No baseline exists for task: {task_id}")

        baseline.promote(
            current_metrics=current_metrics,
            metric_stds=metric_stds,
            sample_size=sample_size,
            reason=f"force:{reason}",
            fingerprint=fingerprint,
            decision_spec=decision_spec,
        )
        baseline.git_commit = self._get_git_commit()

        return baseline

    def list_canary_baselines(self) -> list[str]:
        """List all canary (protected) baseline task IDs."""
        return [
            task_id
            for task_id, baseline in self._baselines.items()
            if baseline.baseline_type == BaselineType.CANARY
        ]

    def list_capability_baselines(self) -> list[str]:
        """List all capability baseline task IDs."""
        return [
            task_id
            for task_id, baseline in self._baselines.items()
            if baseline.baseline_type == BaselineType.CAPABILITY
        ]

    def list_stale_baselines(self) -> list[str]:
        """List all stale baseline task IDs."""
        return [
            task_id
            for task_id, baseline in self._baselines.items()
            if baseline.is_stale
        ]

    def get_baseline_summary(self, task_id: str) -> dict[str, Any] | None:
        """Get a summary of a baseline for reporting."""
        baseline = self.get_baseline(task_id)
        if not baseline:
            return None

        return {
            "task_id": baseline.task_id,
            "type": baseline.baseline_type.value,
            "version": baseline.version,
            "fingerprint": baseline.fingerprint_short,
            "updated_at": baseline.updated_at.isoformat(),
            "is_stale": baseline.is_stale,
            "in_cooldown": baseline.in_cooldown,
            "promotion_count": baseline.promotion_count,
            "metrics": {
                name: {
                    "value": m.baseline_value,
                    "std": m.std_deviation,
                }
                for name, m in baseline.metrics.items()
            },
        }

    def compare_to_baseline_with_ci(
        self,
        task_id: str,
        current_values: dict[str, list[float]],
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
    ) -> dict[str, Any]:
        """Compare current metrics to baseline with bootstrap CI.

        Uses statistical inference to determine if differences are significant.
        This is the research-grade comparison method.

        Args:
            task_id: The task identifier
            current_values: Dict of metric_name -> list of sample values
            confidence: Confidence level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dict of metric_name -> comparison result with:
            - baseline: MetricEstimate for baseline
            - current: MetricEstimate for current
            - delta: point estimate of difference
            - ci_lower, ci_upper: CI bounds for difference
            - is_significant: True if CI doesn't include 0
            - is_regression: True if significant decline
            - cohens_d: Effect size
        """
        baseline = self.get_baseline(task_id)

        if not baseline:
            return {"_no_baseline": True}

        comparisons = {}

        for metric_name, values in current_values.items():
            metric_baseline = baseline.get_metric(metric_name)

            if not metric_baseline:
                continue

            # Estimate current metric with CI
            current_est = estimate_metric(
                values, confidence, n_bootstrap
            )

            # Compare using summary statistics (we don't have raw baseline data)
            comparison = compare_to_baseline_summary(
                baseline_mean=metric_baseline.baseline_value,
                baseline_std=metric_baseline.std_deviation,
                baseline_n=metric_baseline.sample_size,
                current_mean=current_est.mean,
                current_std=current_est.std,
                current_n=current_est.n,
                confidence=confidence,
            )

            # Adjust significance for direction
            is_regression = False
            if comparison.is_significant:
                if metric_baseline.higher_is_better:
                    is_regression = comparison.delta < 0
                else:
                    is_regression = comparison.delta > 0

            comparisons[metric_name] = {
                "baseline": {
                    "mean": metric_baseline.baseline_value,
                    "std": metric_baseline.std_deviation,
                    "n": metric_baseline.sample_size,
                },
                "current": current_est.to_dict(),
                "delta": comparison.delta,
                "relative_delta": comparison.relative_delta,
                "ci_lower": comparison.ci_lower,
                "ci_upper": comparison.ci_upper,
                "confidence": confidence,
                "is_significant": comparison.is_significant,
                "is_regression": is_regression,
                "cohens_d": comparison.cohens_d,
                "effect_magnitude": comparison.effect_magnitude,
                "p_value": comparison.p_value,
                "higher_is_better": metric_baseline.higher_is_better,
            }

        return comparisons
