"""Tests for the CLI baseline-check data path.

RegressionDetector.compare() expects one metric dict per trial so its
t-test sees a real sample distribution. The CLI used to hand it a single
pre-aggregated dict, collapsing the statistics to a one-sample z-test.
"""

from tracelens.baselines.manager import TaskBaseline
from tracelens.cli.main import _per_trial_results
from tracelens.core.outcome import Outcome
from tracelens.core.trial import Trial, TrialBatch, TrialStatus


def _trial(task_id: str, passed: bool, score: float) -> Trial:
    trial = Trial(task_id=task_id, status=TrialStatus.COMPLETED)
    trial.add_outcome(
        Outcome(trial_id="x", grader_id="g", passed=passed, score=score)
    )
    return trial


def _batch(*trials: Trial) -> TrialBatch:
    batch = TrialBatch()
    for trial in trials:
        batch.add_trial(trial)
    return batch


def test_per_trial_results_yields_one_sample_per_trial() -> None:
    batch = _batch(
        _trial("t1", passed=True, score=0.9),
        _trial("t1", passed=True, score=0.8),
        _trial("t1", passed=False, score=0.2),
        _trial("other", passed=True, score=1.0),
    )

    results = _per_trial_results(batch.get_trials_for_task("t1"))

    assert len(results) == 3
    assert [r["pass_rate"] for r in results] == [1.0, 1.0, 0.0]
    assert [r["mean_score"] for r in results] == [0.9, 0.8, 0.2]


def test_detector_sees_distribution_and_flags_regression() -> None:
    baseline = TaskBaseline(task_id="t1")
    baseline.add_metric(metric_name="pass_rate", value=1.0, std=0.05, sample_size=20)
    baseline.add_metric(metric_name="mean_score", value=0.9, std=0.05, sample_size=20)

    batch = _batch(
        *[_trial("t1", passed=False, score=0.1) for _ in range(5)]
    )

    from tracelens.baselines.comparison import RegressionDetector

    report = RegressionDetector().compare(
        baseline, _per_trial_results(batch.get_trials_for_task("t1"))
    )

    assert report.has_regression
    # 5 identical samples → the multi-sample branch ran, not single-sample z
    pass_rate_reg = next(
        r for r in report.regressions if r.metric_name == "pass_rate"
    )
    assert pass_rate_reg.current_mean == 0.0
    assert pass_rate_reg.is_significant


def _infra_trial(task_id: str) -> Trial:
    trial = Trial(task_id=task_id)
    trial.status = TrialStatus.INFRA_ERROR
    trial.error_message = "container OOM"
    return trial


def _grader_error_trial(task_id: str) -> Trial:
    trial = Trial(task_id=task_id)
    trial.status = TrialStatus.COMPLETED
    trial.add_outcome(
        Outcome(
            trial_id="x", grader_id="g", passed=False, score=0.0, grader_error=True
        )
    )
    return trial


def test_per_trial_results_excludes_infra_error_trials() -> None:
    """Infra failures must not masquerade as pass_rate=0.0 agent failures
    in the gate — they are already surfaced via infra_error_rate."""
    trials = [
        _trial("t1", passed=True, score=0.9),
        _trial("t1", passed=True, score=0.8),
        _infra_trial("t1"),
    ]

    results = _per_trial_results(trials)

    assert len(results) == 2
    assert all(r["pass_rate"] == 1.0 for r in results)


def test_per_trial_results_excludes_grader_error_trials() -> None:
    """A broken grading harness must not count as an agent failure."""
    trials = [
        _trial("t1", passed=True, score=0.9),
        _grader_error_trial("t1"),
    ]

    results = _per_trial_results(trials)

    assert len(results) == 1


def test_spec_from_trials_prefers_most_recent_and_warns_on_mixed(
    capsys,
) -> None:
    """On checkpoint resume the oldest trials come first; the current
    run's spec must win, with a visible warning about the mix."""
    from tracelens.cli.main import _spec_from_trials
    from tracelens.core.decision_spec import DecisionSpec, InfraConfig
    from tracelens.core.transcript import Transcript

    def _trial_with_spec(memory_mb: int) -> Trial:
        trial = Trial(task_id="t1")
        transcript = Transcript(task_id="t1", final_output={})
        transcript.decision_spec = DecisionSpec(
            infra=InfraConfig(memory_hard_limit_mb=memory_mb)
        )
        trial.transcript = transcript
        return trial

    old, new = _trial_with_spec(2048), _trial_with_spec(512)

    spec = _spec_from_trials([old, new])

    assert spec is not None
    assert spec.infra is not None
    assert spec.infra.memory_hard_limit_mb == 512
    assert "mixed decision specs" in capsys.readouterr().err
