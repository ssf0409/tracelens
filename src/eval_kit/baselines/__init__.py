"""Baseline management and regression detection."""

from eval_kit.baselines.manager import (
    MetricBaseline,
    TaskBaseline,
    BaselineManager,
    BaselineType,
    PromotionPolicy,
)
from eval_kit.baselines.comparison import (
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
