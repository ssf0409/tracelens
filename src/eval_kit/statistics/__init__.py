"""Statistical analysis for evaluation results."""

from eval_kit.statistics.pass_at_k import (
    pass_at_k,
    pass_at_k_estimator,
    PassAtKAnalyzer,
)
from eval_kit.statistics.consistency import (
    pass_to_k,
    pass_to_k_estimator,
    ConsistencyAnalyzer,
)
from eval_kit.statistics.inference import (
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
