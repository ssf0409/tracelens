"""Tests for explicit metric availability (issue #46)."""

import pytest

from tracelens.statistics.availability import MetricValue, unavailable_reason
from tracelens.statistics.consistency import (
    ConsistencyAnalyzer,
    pass_to_k_estimator,
    pass_to_k_metric,
)
from tracelens.statistics.pass_at_k import (
    PassAtKAnalyzer,
    pass_at_k,
    pass_at_k_estimator,
    pass_at_k_metric,
)


class TestMetricValue:
    def test_available_describe_and_roundtrip(self):
        mv = MetricValue(
            name="pass@1", value=0.85, eligible_tasks=2, total_tasks=2,
            required_runs=1, max_runs=3,
        )
        assert mv.available
        assert mv.describe() == "0.8500 (2/2 tasks)"
        assert mv.to_dict()["available"] is True
        assert MetricValue.from_dict(mv.to_dict()) == mv

    def test_unavailable_describe_and_roundtrip(self):
        mv = MetricValue(
            name="pass@5", value=None, eligible_tasks=0, total_tasks=2,
            required_runs=5, max_runs=1, reason="needs at least 5 gradable runs per task",
        )
        assert not mv.available
        assert mv.describe() == (
            "N/A: needs at least 5 gradable runs per task; "
            "0/2 tasks eligible; max 1 gradable run(s) recorded"
        )
        assert mv.to_dict()["available"] is False
        assert MetricValue.from_dict(mv.to_dict()) == mv

    def test_legacy_entry_has_unknown_counts(self):
        recorded = MetricValue.legacy("pass@3", 0.0)
        assert recorded.available and recorded.total_tasks is None
        assert recorded.describe() == "0.0000"
        missing = MetricValue.legacy("pass^3", None)
        assert missing.reason == "not recorded" and missing.describe() == "N/A: not recorded"

    def test_reason_text(self):
        assert unavailable_reason("gradable runs", 5, 0) == "no tasks with gradable trials"
        assert unavailable_reason("gradable runs", 5, 3) == "needs at least 5 gradable runs per task"


class TestPassAtKMetric:
    def test_empty_input(self):
        mv = pass_at_k_metric({}, k=1)
        assert mv.value is None
        assert (mv.eligible_tasks, mv.total_tasks, mv.max_runs) == (0, 0, 0)
        assert mv.reason == "no tasks with gradable trials"

    def test_n_below_k_is_unavailable_not_estimated(self):
        results = {"a": [True, True], "b": [True]}
        mv = pass_at_k_metric(results, k=3)
        assert mv.value is None
        assert (mv.eligible_tasks, mv.total_tasks, mv.max_runs, mv.required_runs) == (0, 2, 2, 3)
        assert mv.reason == "needs at least 3 gradable runs per task"
        # Float API: legacy 0.0 placeholder, no c/n fallback.
        assert pass_at_k_estimator(results, k=3) == 0.0

    def test_n_equal_to_k_and_above(self):
        assert pass_at_k_metric({"a": [True, False, True]}, k=3).value == pytest.approx(
            pass_at_k(3, 2, 3)
        )
        assert pass_at_k_metric({"a": [True, False, True, True]}, k=3).value == pytest.approx(
            pass_at_k(4, 3, 3)
        )

    def test_mixed_sample_counts_report_the_eligible_subset(self):
        results = {"a": [True, False, False], "b": [False]}
        mv = pass_at_k_metric(results, k=2)
        assert mv.value == pytest.approx(pass_at_k(3, 1, 2))
        assert (mv.eligible_tasks, mv.total_tasks, mv.max_runs) == (1, 2, 3)
        assert pass_at_k_estimator(results, k=2) == pytest.approx(mv.value)

    def test_all_pass_and_all_fail_are_measured_values(self):
        assert pass_at_k_metric({"a": [True] * 3, "b": [True] * 3}, k=2).value == 1.0
        zero = pass_at_k_metric({"a": [False] * 3}, k=2)
        assert zero.value == 0.0 and zero.available

    def test_task_without_gradable_runs_counts_in_total_only(self):
        mv = pass_at_k_metric({"a": [True], "excluded": []}, k=1)
        assert mv.value == 1.0
        assert (mv.eligible_tasks, mv.total_tasks) == (1, 2)

    def test_k_below_one_raises(self):
        with pytest.raises(ValueError, match="k must be at least 1"):
            pass_at_k_metric({"a": [True]}, k=0)

    def test_analyzer_detailed_agrees_with_float_api_when_available(self):
        results = {"a": [True, False, True], "b": [True, True, True]}
        analyzer = PassAtKAnalyzer(k_values=[1, 3, 5])
        detailed = analyzer.analyze_detailed(results)
        floats = analyzer.analyze(results)
        assert detailed["pass@1"].value == pytest.approx(floats["pass@1"])
        assert detailed["pass@3"].value == pytest.approx(floats["pass@3"])
        assert detailed["pass@5"].value is None and floats["pass@5"] == 0.0


class TestPassToKMetric:
    def test_empty_input(self):
        mv = pass_to_k_metric({}, k=2)
        assert mv.value is None and mv.reason == "no tasks with gradable trials"

    def test_no_complete_window_is_unavailable(self):
        mv = pass_to_k_metric({"a": [True, None, True], "b": [True]}, k=2)
        assert mv.value is None
        assert (mv.eligible_tasks, mv.total_tasks, mv.max_runs) == (0, 2, 2)
        assert mv.reason == "needs at least 2 consecutive gradable runs per task"
        assert pass_to_k_estimator({"a": [True, None, True]}, k=2) == 0.0

    def test_n_equal_to_k_and_above(self):
        assert pass_to_k_metric({"a": [True, True]}, k=2).value == 1.0
        assert pass_to_k_metric({"a": [True, True, False]}, k=2).value == 0.5

    def test_mixed_sample_counts_report_the_eligible_subset(self):
        mv = pass_to_k_metric({"a": [True, True, False], "b": [True]}, k=2)
        assert mv.value == 0.5
        assert (mv.eligible_tasks, mv.total_tasks, mv.max_runs) == (1, 2, 3)

    def test_all_fail_is_a_measured_zero(self):
        mv = pass_to_k_metric({"a": [False, False, False]}, k=2)
        assert mv.value == 0.0 and mv.available

    def test_analyzer_detailed(self):
        analyzer = ConsistencyAnalyzer(k_values=[2, 5])
        detailed = analyzer.analyze_detailed({"a": [True, True, False]})
        assert detailed["pass^2"].value == 0.5
        assert detailed["pass^5"].value is None and detailed["pass^5"].required_runs == 5
