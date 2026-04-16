"""Baseline management and regression detection."""

from eval_kit.baselines.comparison import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
)
from eval_kit.baselines.manager import (
    BaselineManager,
    BaselineType,
    MetricBaseline,
    PromotionPolicy,
    TaskBaseline,
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
