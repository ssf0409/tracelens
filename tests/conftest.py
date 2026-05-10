"""Shared test fixtures and configuration."""

from pathlib import Path

import pytest

from eval_kit.baselines.manager import TaskBaseline
from eval_kit.core._time import utc_now
from eval_kit.core.outcome import Outcome
from eval_kit.core.task import Task, TaskExpectation
from eval_kit.core.transcript import StepType, ToolCall, Transcript, TranscriptStep
from eval_kit.core.trial import Trial, TrialStatus


@pytest.fixture
def sample_task() -> Task:
    """Create a sample task for testing."""
    return Task(
        task_id="test-task-001",
        name="Test Goal Decomposition",
        description="Decompose a test goal into phases and tasks",
        input_data={
            "goal": "Learn Python programming",
            "user_context": {
                "experience": "beginner",
                "hours_per_week": 10,
                "budget": "free",
            },
        },
        expectation=TaskExpectation(
            expected_metrics={"quality": 0.8},
            metric_thresholds={"quality": 0.7},
        ),
        category="programming",
        tags=["python", "beginner"],
        difficulty="medium",
    )


@pytest.fixture
def sample_transcript(sample_task: Task) -> Transcript:
    """Create a sample transcript for testing."""
    transcript = Transcript(
        transcript_id="test-transcript-001",
        task_id=sample_task.task_id,
        agent_name="test_agent",
        agent_version="1.0.0",
        started_at=utc_now(),
    )

    # Add some steps
    transcript.add_step(TranscriptStep(
        step_id="step-1",
        step_type=StepType.LLM_CALL,
        content="Analyzing goal...",
        model="gpt-4",
        tokens_in=100,
        tokens_out=50,
    ))

    transcript.add_step(TranscriptStep(
        step_id="step-2",
        step_type=StepType.TOOL_CALL,
        tool_call=ToolCall(
            tool_name="search",
            arguments={"query": "python tutorials"},
            result=["tutorial1", "tutorial2"],
            duration_ms=150.0,
        ),
    ))

    transcript.add_step(TranscriptStep(
        step_id="step-3",
        step_type=StepType.AGENT_OUTPUT,
        content={"phases": ["Learn basics", "Build projects"]},
    ))

    transcript.final_output = {
        "phases": [
            {"name": "Phase 1", "tasks": ["Install Python", "Read tutorial"]},
            {"name": "Phase 2", "tasks": ["Build calculator", "Build game"]},
        ],
        "quality_score": 0.85,
    }
    transcript.completed_at = utc_now()

    return transcript


@pytest.fixture
def sample_outcome() -> Outcome:
    """Create a sample outcome for testing."""
    return Outcome(
        outcome_id="test-outcome-001",
        trial_id="test-trial-001",
        grader_id="quality_grader",
        passed=True,
        score=0.85,
        metrics={
            "specificity": 0.9,
            "personalization": 0.8,
            "actionability": 0.85,
        },
        feedback="Good decomposition with specific tasks",
    )


@pytest.fixture
def sample_trial(sample_task: Task, sample_transcript: Transcript, sample_outcome: Outcome) -> Trial:
    """Create a sample trial for testing."""
    trial = Trial(
        trial_id="test-trial-001",
        task_id=sample_task.task_id,
        run_index=0,
        total_runs=5,
        status=TrialStatus.COMPLETED,
        transcript=sample_transcript,
        started_at=sample_transcript.started_at,
        completed_at=sample_transcript.completed_at,
    )
    trial.add_outcome(sample_outcome)
    return trial


@pytest.fixture
def sample_baseline() -> TaskBaseline:
    """Create a sample baseline for testing."""
    baseline = TaskBaseline(
        task_id="btc_backtest",
        task_name="BTC Daily Backtest",
        version="1.0.0",
        environment="test",
        git_commit="abc123",
    )

    baseline.add_metric(
        metric_name="sharpe_ratio",
        value=1.2,
        std=0.3,
        sample_size=100,
        absolute_threshold=-0.2,
        relative_threshold=0.10,
        higher_is_better=True,
    )

    baseline.add_metric(
        metric_name="max_drawdown",
        value=-0.15,
        std=0.05,
        sample_size=100,
        absolute_threshold=-0.05,
        relative_threshold=0.20,
        higher_is_better=True,  # Higher (less negative) is better for drawdown
    )

    baseline.add_metric(
        metric_name="win_rate",
        value=0.55,
        std=0.05,
        sample_size=100,
        absolute_threshold=-0.05,
        relative_threshold=0.10,
        higher_is_better=True,
    )

    return baseline


@pytest.fixture
def temp_baselines_file(tmp_path: Path) -> Path:
    """Create a temporary baselines file path."""
    return tmp_path / "baselines.json"


@pytest.fixture
def pass_results_by_task() -> dict[str, list[bool]]:
    """Create sample pass results for statistics tests."""
    return {
        "task1": [True, True, True, False, True, True, False, True, True, True],
        "task2": [True, False, True, True, False, True, True, True, False, True],
        "task3": [True, True, True, True, True, True, True, True, True, True],
        "task4": [False, False, True, False, False, True, False, False, True, False],
    }
