"""Baseline management and regression detection."""

from agent_eval.baselines.manager import (
    MetricBaseline,
    TaskBaseline,
    BaselineManager,
)
from agent_eval.baselines.comparison import (
    RegressionSeverity,
    MetricRegression,
    RegressionReport,
    RegressionDetector,
)

__all__ = [
    "MetricBaseline",
    "TaskBaseline",
    "BaselineManager",
    "RegressionSeverity",
    "MetricRegression",
    "RegressionReport",
    "RegressionDetector",
]
