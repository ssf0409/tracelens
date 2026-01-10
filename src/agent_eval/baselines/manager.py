"""Baseline storage and retrieval.

Baselines store historical performance metrics that are used to
detect regressions when new evaluations are run.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import subprocess

from pydantic import BaseModel, Field


class MetricBaseline(BaseModel):
    """Baseline for a single metric.

    Stores the expected value, standard deviation, and thresholds
    for regression detection.
    """

    metric_name: str
    baseline_value: float
    std_deviation: float = 0.0
    sample_size: int = 1
    computed_at: datetime = Field(default_factory=datetime.utcnow)

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
    """

    task_id: str
    task_name: str | None = None
    version: str = "1.0.0"
    environment: str = "development"

    # Metric baselines
    metrics: dict[str, MetricBaseline] = Field(default_factory=dict)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = "manual"  # 'manual', 'ci', 'scheduled'
    git_commit: str | None = None
    git_branch: str | None = None

    # History for tracking changes
    previous_versions: list[dict[str, Any]] = Field(default_factory=list)

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
        self.updated_at = datetime.utcnow()

    def get_metric(self, metric_name: str) -> MetricBaseline | None:
        """Get a specific metric baseline."""
        return self.metrics.get(metric_name)


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

        return TaskBaseline(
            task_id=data["task_id"],
            task_name=data.get("task_name"),
            version=data.get("version", "1.0.0"),
            environment=data.get("environment", "development"),
            metrics=metrics,
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data else datetime.utcnow(),
            created_by=data.get("created_by", "manual"),
            git_commit=data.get("git_commit"),
            git_branch=data.get("git_branch"),
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
    ) -> TaskBaseline:
        """Update or create a baseline with new metric values.

        Args:
            task_id: The task identifier
            metrics: Dict of metric_name -> value
            metric_stds: Optional dict of metric_name -> std deviation
            sample_size: Number of samples used to compute metrics
            keep_thresholds: Keep existing thresholds when updating

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

        existing.updated_at = datetime.utcnow()
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
        except Exception:
            pass
        return None

    def list_tasks(self) -> list[str]:
        """List all task IDs with baselines."""
        return list(self._baselines.keys())

    def compare_to_baseline(
        self,
        task_id: str,
        current_metrics: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
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
