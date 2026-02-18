"""Statistical analysis for evaluation results."""

from agent_eval.statistics.pass_at_k import (
    pass_at_k,
    pass_at_k_estimator,
    PassAtKAnalyzer,
)
from agent_eval.statistics.consistency import (
    pass_to_k,
    pass_to_k_estimator,
    ConsistencyAnalyzer,
)
from agent_eval.statistics.inference import (
    MetricEstimate,
    ComparisonResult,
    bootstrap_ci,
    estimate_metric,
    bootstrap_difference_ci,
    cohens_d,
    permutation_test,
    compare_metrics,
    compare_to_baseline_summary,
)

__all__ = [
    # pass@k (capability)
    "pass_at_k",
    "pass_at_k_estimator",
    "PassAtKAnalyzer",
    # pass^k (reliability)
    "pass_to_k",
    "pass_to_k_estimator",
    "ConsistencyAnalyzer",
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
]
