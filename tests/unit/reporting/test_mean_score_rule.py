"""One ``mean_score`` rule for the report, the baseline gate and ``compare`` (#114).

A gradable trial without a score (a timeout, or an agent failure before any
grader ran) is excluded from the mean and counted; ``pass_rate`` already
records it as a failure. The gate used to impute 0.0 instead, so a baseline
stored from the report's task summaries blocked a run whose agent had not
changed at all.
"""

import json
import math
from pathlib import Path

import pytest

from tracelens.baselines.manager import BaselineManager, TaskBaseline
from tracelens.core.outcome import Outcome
from tracelens.core.trial import Trial, TrialBatch, TrialStatus
from tracelens.reporting.gate import GateStatus, evaluate_gate, per_trial_results
from tracelens.reporting.generator import ReportData, ReportGenerator
from tracelens.statistics.run_comparison import MetricSelector


def _scored(task_id: str, run_index: int, score: float, *, passed: bool = True) -> Trial:
    trial = Trial(task_id=task_id, run_index=run_index, status=TrialStatus.COMPLETED)
    trial.add_outcome(
        Outcome(trial_id=trial.trial_id, grader_id="g", passed=passed, score=score)
    )
    return trial


def _run(*, scored: int, timeouts: int, score: float = 0.9, task_id: str = "t") -> TrialBatch:
    """``scored`` graded trials at ``score``, then ``timeouts`` trials that never got a score."""
    batch = TrialBatch()
    for i in range(scored):
        batch.add_trial(_scored(task_id, i, score))
    for i in range(scored, scored + timeouts):
        batch.add_trial(Trial(task_id=task_id, run_index=i, status=TrialStatus.TIMEOUT))
    return batch


def test_report_gate_and_compare_agree_on_a_task_with_a_timeout() -> None:
    batch = _run(scored=4, timeouts=1)

    summary = ReportGenerator().build_report(batch).task_summaries[0]
    gate_samples = per_trial_results(batch.trials)
    gate_scores = [r["mean_score"] for r in gate_samples if "mean_score" in r]
    selector = MetricSelector.parse("mean_score")
    compare_scores = [v for t in batch.trials if (v := selector.value(t)) is not None]

    assert summary.mean_score == pytest.approx(0.9)
    assert sum(gate_scores) / len(gate_scores) == pytest.approx(0.9)
    assert sum(compare_scores) / len(compare_scores) == pytest.approx(0.9)
    # Excluded and counted, in every place.
    assert (summary.scored_trials, summary.gradable_trials) == (4, 5)
    assert len(gate_scores) == len(compare_scores) == 4
    # The timeout still counts against pass_rate: it is a failure, not a gap.
    assert summary.pass_rate == pytest.approx(0.8)
    assert [r["pass_rate"] for r in gate_samples] == [1.0, 1.0, 1.0, 1.0, 0.0]


def _store_from_report(report: ReportData, path: Path) -> BaselineManager:
    """The documented recipe: baselines from the results file's task summaries."""
    manager = BaselineManager(path)
    for task in report.to_dict()["task_summaries"]:
        baseline = TaskBaseline(task_id=task["task_id"])
        baseline.add_metric(
            "pass_rate", task["pass_rate"], std=0.0,
            sample_size=task["gradable_trials"], is_rate=True,
        )
        if task["mean_score"] is not None:
            baseline.add_metric(
                "mean_score", task["mean_score"], std=task["std_score"],
                sample_size=task.get("scored_trials", task["gradable_trials"]),
            )
        manager.set_baseline(baseline)
    return manager


def test_a_baseline_stored_from_the_report_does_not_flip_on_the_definition(
    tmp_path: Path,
) -> None:
    """The issue's repro: 0.9 scores with one run in five timing out, the same agent twice.

    Imputing 0.0 read the unchanged agent as ``0.9000 -> 0.7200`` SEVERE
    while the report beside it printed 0.9000; at 40 runs a side that drop
    is significant (p = 0.021) and the gate blocked.
    """
    reference = ReportGenerator().build_report(_run(scored=32, timeouts=8))
    manager = _store_from_report(reference, tmp_path / "baselines.json")

    gate = evaluate_gate(_run(scored=32, timeouts=8), manager)

    task = gate.tasks[0]
    assert task.compared_metrics == ["mean_score", "pass_rate"]
    assert task.regressions == [] and not task.blocking
    assert gate.status is GateStatus.PASSED


def test_a_real_drop_in_the_scored_trials_is_still_caught(tmp_path: Path) -> None:
    reference = ReportGenerator().build_report(_run(scored=16, timeouts=4))
    manager = _store_from_report(reference, tmp_path / "baselines.json")

    worse = TrialBatch()
    for i in range(16):
        worse.add_trial(_scored("t", i, 0.5 + 0.01 * (i % 4)))
    for i in range(16, 20):
        worse.add_trial(Trial(task_id="t", run_index=i, status=TrialStatus.TIMEOUT))
    gate = evaluate_gate(worse, manager)

    (finding,) = gate.tasks[0].regressions
    assert finding.metric_name == "mean_score"
    assert finding.current_mean == pytest.approx(0.515)
    assert finding.baseline_mean == pytest.approx(0.9)


def test_a_task_with_no_scored_trial_has_no_mean_score(tmp_path: Path) -> None:
    batch = _run(scored=0, timeouts=3)
    generator = ReportGenerator()
    report = generator.build_report(batch)

    summary = report.task_summaries[0]
    assert summary.mean_score is None and summary.scored_trials == 0
    assert report.overall_mean_score is None and report.scored_trials == 0
    data = json.loads(json.dumps(report.to_dict()))
    assert data["overall_mean_score"] is None
    assert data["task_summaries"][0]["mean_score"] is None
    assert ReportData.from_dict(data).task_summaries[0].mean_score is None

    markdown = generator.render_markdown(report)
    assert "- **Mean Score**: n/a (no scored trials)" in markdown
    assert "| t | 3 | 0.0% | n/a |" in markdown
    assert "mean_score=n/a" in generator.render_ci_summary(report)
    html = generator.render_html(report)
    assert "<td>n/a</td><td>n/a</td></tr>" in html
    # The gate compares what the run measured: pass_rate, not an invented score.
    assert [r for r in per_trial_results(batch.trials)] == [{"pass_rate": 0.0}] * 3


def test_reports_show_the_excluded_count_next_to_the_mean() -> None:
    generator = ReportGenerator()
    report = generator.build_report(_run(scored=4, timeouts=1))

    markdown = generator.render_markdown(report)
    assert (
        "- **Mean Score**: 0.9000 (over 4 scored trials; 1 gradable trial(s) "
        "without a score excluded)"
    ) in markdown
    assert "| t | 5 | 80.0% | 0.9000 (1 unscored) |" in markdown
    assert "0.9000 (1 unscored)" in generator.render_html(report)
    assert "mean_score=0.9000" in generator.render_ci_summary(report)


def test_a_fully_scored_run_reads_as_before() -> None:
    generator = ReportGenerator()
    report = generator.build_report(_run(scored=3, timeouts=0))

    markdown = generator.render_markdown(report)
    assert "- **Mean Score**: 0.9000\n" in markdown
    assert "| t | 3 | 100.0% | 0.9000 |" in markdown


def test_a_legacy_report_loads_and_renders_its_recorded_mean() -> None:
    legacy = {
        "total_trials": 2,
        "total_tasks": 1,
        "gradable_trials": 2,
        "overall_pass_rate": 0.5,
        "overall_mean_score": 0.0,
        "task_summaries": [{
            "task_id": "t", "num_trials": 2, "gradable_trials": 2,
            "pass_rate": 0.5, "mean_score": 0.0, "std_score": 0.0,
        }],
    }
    report = ReportData.from_dict(legacy)

    assert report.scored_trials is None and report.task_summaries[0].scored_trials is None
    markdown = ReportGenerator().render_markdown(report)
    assert "- **Mean Score**: 0.0000\n" in markdown
    assert "| t | 2 | 50.0% | 0.0000 |" in markdown


def _nan_scored() -> Trial:
    """A score that bypassed validation (``Outcome`` rejects NaN on construction)."""
    trial = Trial(task_id="t", status=TrialStatus.COMPLETED)
    trial.outcomes.append(
        Outcome.model_construct(trial_id=trial.trial_id, grader_id="g", passed=True, score=math.nan)
    )
    return trial


@pytest.mark.parametrize(
    ("trial", "expected"),
    [
        (_scored("t", 0, 0.75), 0.75),
        (_scored("t", 0, 0.0, passed=False), 0.0),
        (_nan_scored(), None),
        (Trial(task_id="t", status=TrialStatus.TIMEOUT), None),
        (Trial(task_id="t", status=TrialStatus.INFRA_ERROR), None),
        (Trial(task_id="t", status=TrialStatus.PENDING), None),
    ],
    ids=["scored", "scored-zero", "nan", "timeout", "infra-error", "never-run"],
)
def test_mean_score_value(trial: Trial, expected: float | None) -> None:
    assert trial.mean_score_value == expected


def _baseline(tmp_path: Path, **metrics: tuple[float, int]) -> BaselineManager:
    manager = BaselineManager(tmp_path / "baselines.json")
    baseline = TaskBaseline(task_id="t")
    for name, (value, n) in metrics.items():
        baseline.add_metric(name, value, std=0.1, sample_size=n, is_rate=name == "pass_rate")
    manager.set_baseline(baseline)
    return manager


def test_a_stored_mean_score_the_run_did_not_measure_is_never_a_pass(tmp_path: Path) -> None:
    """Task t always failed but was graded; now every run of it times out.

    Its pass_rate cannot move (0/10 -> 0/10) and none of its trials has a
    score. With a healthy task u beside it, leaving mean_score out of t's
    comparison passed the gate with exit 0 (as did v0.7.0's imputed zeros).
    """
    manager = BaselineManager(tmp_path / "baselines.json")
    for task_id, pass_rate, mean_score in (("t", 0.0, 0.4), ("u", 1.0, 0.9)):
        baseline = TaskBaseline(task_id=task_id)
        baseline.add_metric("pass_rate", pass_rate, std=0.1, sample_size=10, is_rate=True)
        baseline.add_metric("mean_score", mean_score, std=0.1, sample_size=10)
        manager.set_baseline(baseline)
    batch = _run(scored=0, timeouts=10)
    for i in range(10):
        batch.add_trial(_scored("u", i, 0.9))

    gate = evaluate_gate(batch, manager)

    t = next(task for task in gate.tasks if task.task_id == "t")
    assert t.compared_metrics == ["pass_rate"]
    assert t.unmeasured_metrics == ["mean_score"] and t.unscored_trials == 10
    assert gate.status is GateStatus.UNEVALUABLE and gate.exit_code == 2
    assert (
        "1 task(s) with a stored metric the run did not measure: t (mean_score: "
        "none of the 10 gradable trial(s) had a score)"
    ) in gate.reasons


def test_a_mean_score_only_baseline_names_the_missing_scores(tmp_path: Path) -> None:
    manager = _baseline(tmp_path, mean_score=(0.9, 10))

    gate = evaluate_gate(_run(scored=0, timeouts=5), manager)

    task = gate.tasks[0]
    assert task.outcome.value == "no_comparable_metrics"
    assert task.reason == (
        "the run measured none of the baseline's metrics: "
        "mean_score: none of the 5 gradable trial(s) had a score"
    )
    assert gate.status is GateStatus.UNEVALUABLE


def test_the_gate_records_the_unscored_count_and_round_trips_it(tmp_path: Path) -> None:
    manager = _baseline(tmp_path, pass_rate=(0.8, 40), mean_score=(0.9, 32))

    gate = evaluate_gate(_run(scored=32, timeouts=8), manager)

    task = gate.tasks[0]
    assert task.unscored_trials == 8 and task.unmeasured_metrics == []
    assert gate.status is GateStatus.PASSED
    data = json.loads(json.dumps(gate.to_dict()))
    assert data["tasks"][0]["unscored_trials"] == 8
    assert data["tasks"][0]["unmeasured_metrics"] == []
    restored = type(gate).from_dict(data).tasks[0]
    assert (restored.unscored_trials, restored.unmeasured_metrics) == (8, [])


def test_the_ci_line_and_the_html_card_carry_the_unscored_count() -> None:
    generator = ReportGenerator()
    report = generator.build_report(_run(scored=4, timeouts=4))

    assert report.unscored_trials == 4
    assert "mean_score=0.9000, unscored_trials=4" in generator.render_ci_summary(report)
    assert '<div class="label">Mean Score (4 unscored)</div>' in generator.render_html(report)
    full = generator.build_report(_run(scored=3, timeouts=0))
    assert "unscored" not in generator.render_ci_summary(full)


def test_a_task_with_no_scored_trial_gets_no_mean_score_baseline(tmp_path: Path) -> None:
    batch = _run(scored=0, timeouts=3)
    for i in range(4):
        batch.add_trial(_scored("u", i, 0.7))
    report = ReportGenerator().build_report(batch)

    manager = _store_from_report(report, tmp_path / "baselines.json")

    unscored = manager.get_baseline("t")
    scored = manager.get_baseline("u")
    assert unscored is not None and scored is not None
    assert set(unscored.metrics) == {"pass_rate"}
    assert scored.metrics["mean_score"].sample_size == 4


def test_a_measured_zero_mean_is_drawn_in_the_score_histogram() -> None:
    batch = TrialBatch()
    for i in range(3):
        batch.add_trial(_scored("t", i, 0.0, passed=False))
    html = ReportGenerator().render_html(ReportGenerator().build_report(batch))

    assert "<h2>Score Distribution</h2>" in html


def test_a_task_that_blocks_on_pass_rate_still_blocks_without_scores(tmp_path: Path) -> None:
    """Agent failures are regression evidence: a total collapse blocks (exit 1)."""
    manager = _baseline(tmp_path, pass_rate=(1.0, 20), mean_score=(0.9, 20))

    gate = evaluate_gate(_run(scored=0, timeouts=10), manager)

    task = gate.tasks[0]
    assert task.unmeasured_metrics == ["mean_score"] and task.blocking
    assert gate.status is GateStatus.BLOCKED and gate.exit_code == 1
