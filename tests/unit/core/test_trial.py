"""Tests for trial module."""


import pytest

from eval_kit.core.outcome import Outcome
from eval_kit.core.trial import Trial, TrialBatch, TrialStatus


class TestTrialStatus:
    """Tests for TrialStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert TrialStatus.PENDING == "pending"
        assert TrialStatus.RUNNING == "running"
        assert TrialStatus.COMPLETED == "completed"
        assert TrialStatus.FAILED == "failed"
        assert TrialStatus.TIMEOUT == "timeout"
        assert TrialStatus.SKIPPED == "skipped"


class TestTrial:
    """Tests for Trial model."""

    def test_trial_creation_minimal(self):
        """Test creating a trial with minimal fields."""
        trial = Trial(task_id="task-001")

        assert trial.task_id == "task-001"
        assert trial.trial_id  # Auto-generated
        assert trial.run_index == 0
        assert trial.total_runs == 1
        assert trial.status == TrialStatus.PENDING
        assert trial.outcomes == []

    def test_trial_creation_full(self, sample_trial: Trial):
        """Test creating a full trial."""
        assert sample_trial.task_id == "test-task-001"
        assert sample_trial.run_index == 0
        assert sample_trial.total_runs == 5
        assert sample_trial.status == TrialStatus.COMPLETED
        assert sample_trial.transcript is not None
        assert len(sample_trial.outcomes) == 1

    def test_trial_passed(self):
        """Test passed property."""
        trial = Trial(task_id="task-001")

        # No outcomes = not passed
        assert trial.passed is False

        # Add passing outcome
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id,
            grader_id="g1",
            passed=True,
            score=0.9,
        ))
        assert trial.passed is True

        # Add failing outcome = not passed (all must pass)
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id,
            grader_id="g2",
            passed=False,
            score=0.4,
        ))
        assert trial.passed is False

    def test_trial_aggregate_score(self):
        """Test aggregate score calculation."""
        trial = Trial(task_id="task-001")

        assert trial.aggregate_score is None

        trial.add_outcome(Outcome(
            trial_id=trial.trial_id,
            grader_id="g1",
            passed=True,
            score=0.9,
        ))
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id,
            grader_id="g2",
            passed=True,
            score=0.7,
        ))

        assert trial.aggregate_score == pytest.approx(0.8, rel=0.01)

    def test_trial_is_complete(self):
        """Test is_complete property."""
        trial = Trial(task_id="task-001")

        trial.status = TrialStatus.PENDING
        assert trial.is_complete is False

        trial.status = TrialStatus.RUNNING
        assert trial.is_complete is False

        trial.status = TrialStatus.COMPLETED
        assert trial.is_complete is True

        trial.status = TrialStatus.FAILED
        assert trial.is_complete is True

        trial.status = TrialStatus.TIMEOUT
        assert trial.is_complete is True

    def test_trial_is_successful(self):
        """Test is_successful property."""
        trial = Trial(task_id="task-001")

        trial.status = TrialStatus.COMPLETED
        assert trial.is_successful is True

        trial.error_message = "Something failed"
        assert trial.is_successful is False

        trial.error_message = None
        trial.status = TrialStatus.FAILED
        assert trial.is_successful is False

    def test_trial_duration(self, sample_trial: Trial):
        """Test duration calculation."""
        duration = sample_trial.duration_ms

        assert duration is not None
        assert duration >= 0

    def test_trial_add_outcome(self):
        """Test adding outcomes."""
        trial = Trial(task_id="task-001")

        outcome = Outcome(
            trial_id="different-id",  # Will be overwritten
            grader_id="g1",
            passed=True,
            score=0.9,
        )
        trial.add_outcome(outcome)

        assert len(trial.outcomes) == 1
        assert trial.outcomes[0].trial_id == trial.trial_id

    def test_trial_get_outcome_by_grader(self, sample_trial: Trial):
        """Test getting outcome by grader ID."""
        outcome = sample_trial.get_outcome_by_grader("quality_grader")
        assert outcome is not None
        assert outcome.grader_id == "quality_grader"

        outcome = sample_trial.get_outcome_by_grader("nonexistent")
        assert outcome is None

    def test_trial_get_metric(self, sample_trial: Trial):
        """Test getting a metric value."""
        value = sample_trial.get_metric("specificity")
        assert value == 0.9

        value = sample_trial.get_metric("nonexistent")
        assert value is None

    def test_trial_to_summary_dict(self, sample_trial: Trial):
        """Test trial summary."""
        summary = sample_trial.to_summary_dict()

        assert summary["task_id"] == "test-task-001"
        assert summary["status"] == "completed"
        assert summary["passed"] is True
        assert "outcomes" in summary

    def test_trial_to_ci_dict(self, sample_trial: Trial):
        """Test CI output format."""
        ci_dict = sample_trial.to_ci_dict()

        assert ci_dict["task"] == "test-task-001"
        assert ci_dict["run"] == "1/5"
        assert ci_dict["passed"] is True


class TestTrialBatch:
    """Tests for TrialBatch model."""

    def test_batch_creation(self):
        """Test creating a batch."""
        batch = TrialBatch()

        assert batch.batch_id  # Auto-generated
        assert batch.trials == []
        assert batch.total_count == 0

    def test_batch_with_trials(self):
        """Test batch with trials."""
        trials = [
            Trial(task_id="t1", status=TrialStatus.COMPLETED),
            Trial(task_id="t2", status=TrialStatus.COMPLETED),
            Trial(task_id="t3", status=TrialStatus.RUNNING),
        ]

        # Add passing outcomes to first two
        trials[0].add_outcome(Outcome(
            trial_id=trials[0].trial_id,
            grader_id="g1",
            passed=True,
            score=0.9,
        ))
        trials[1].add_outcome(Outcome(
            trial_id=trials[1].trial_id,
            grader_id="g1",
            passed=False,
            score=0.4,
        ))

        batch = TrialBatch(trials=trials)

        assert batch.total_count == 3
        assert batch.completed_count == 2
        assert batch.passed_count == 1
        assert batch.pass_rate == pytest.approx(1 / 3, rel=0.01)
        assert batch.all_complete is False

    def test_batch_get_trials_for_task(self):
        """Test getting trials for a specific task."""
        trials = [
            Trial(task_id="t1", run_index=0),
            Trial(task_id="t1", run_index=1),
            Trial(task_id="t2", run_index=0),
        ]

        batch = TrialBatch(trials=trials)

        t1_trials = batch.get_trials_for_task("t1")
        assert len(t1_trials) == 2

        t2_trials = batch.get_trials_for_task("t2")
        assert len(t2_trials) == 1

    def test_batch_get_pass_results_by_task(self):
        """Test getting pass results grouped by task."""
        trials = [
            Trial(task_id="t1", run_index=0),
            Trial(task_id="t1", run_index=1),
            Trial(task_id="t2", run_index=0),
        ]

        # Add outcomes
        trials[0].add_outcome(Outcome(
            trial_id=trials[0].trial_id,
            grader_id="g1",
            passed=True,
            score=0.9,
        ))
        trials[1].add_outcome(Outcome(
            trial_id=trials[1].trial_id,
            grader_id="g1",
            passed=False,
            score=0.4,
        ))
        trials[2].add_outcome(Outcome(
            trial_id=trials[2].trial_id,
            grader_id="g1",
            passed=True,
            score=0.8,
        ))

        batch = TrialBatch(trials=trials)
        results = batch.get_pass_results_by_task()

        assert "t1" in results
        assert results["t1"] == [True, False]
        assert "t2" in results
        assert results["t2"] == [True]

    def test_batch_add_trial(self):
        """Test adding trials to a batch via add_trial()."""
        batch = TrialBatch()
        assert batch.total_count == 0

        trial = Trial(task_id="t1", run_index=0)
        batch.add_trial(trial)
        assert batch.total_count == 1
        assert batch.trials[0].task_id == "t1"
