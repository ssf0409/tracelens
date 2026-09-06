"""Statistical analysis for evaluation results."""

from tracelens.statistics.availability import MetricValue
from tracelens.statistics.consistency import (
    ConsistencyAnalyzer,
    pass_to_k,
    pass_to_k_estimator,
    pass_to_k_metric,
)
from tracelens.statistics.inference import (
    ComparisonResult,
    MetricEstimate,
    bootstrap_ci,
    bootstrap_difference_ci,
    cohens_d,
    compare_metrics,
    compare_to_baseline_summary,
    estimate_metric,
    permutation_test,
)
from tracelens.statistics.latency import (
    AggregateLatencyMetrics,
    LatencyAnalyzer,
    LatencyMetrics,
)
from tracelens.statistics.pass_at_k import (
    PassAtKAnalyzer,
    pass_at_k,
    pass_at_k_estimator,
    pass_at_k_metric,
)

__all__ = [
    # pass@k (capability)
    "pass_at_k",
    "pass_at_k_estimator",
    "pass_at_k_metric",
    "PassAtKAnalyzer",
    # pass^k (reliability)
    "pass_to_k",
    "pass_to_k_estimator",
    "pass_to_k_metric",
    "ConsistencyAnalyzer",
    # Metric availability
    "MetricValue",
    # Statistical inference
    "MetricEstimate",
    "ComparisonResult",
    "bootstrap_ci",
    "estimate_metric",
    "bootstrap_difference_ci",
    "cohens_d",
    "permutation_test",
    "compare_metrics",
    "compare_to_baseline_summary",
    # Latency analysis
    "LatencyAnalyzer",
    "LatencyMetrics",
    "AggregateLatencyMetrics",
]
