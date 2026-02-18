"""Baseline management and regression detection."""

from agent_eval.baselines.manager import (
    MetricBaseline,
    TaskBaseline,
    BaselineManager,
    BaselineType,
    PromotionPolicy,
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
    "BaselineType",
    "PromotionPolicy",
    "RegressionSeverity",
    "MetricRegression",
    "RegressionReport",
    "RegressionDetector",
]
