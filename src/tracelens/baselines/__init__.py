"""Baseline management and regression detection."""

from tracelens.baselines.comparison import (
    MetricRegression,
    RegressionDetector,
    RegressionReport,
    RegressionSeverity,
)
from tracelens.baselines.manager import (
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
