"""Tests for result persistence: TrialBatch and Transcript round-trip serialization."""

import json
from datetime import datetime, timezone

import pytest

from eval_kit.core.decision_spec import (
    AgentSpec,
    DecisionSpec,
    EnvironmentSpec,
    ModelConfig,
    PromptSpec,
    ToolSpec,
)
from eval_kit.core.outcome import Outcome
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript, TranscriptStep, StepType, ToolCall
from eval_kit.core.trial import Trial, TrialBatch, TrialStatus


class TestTranscriptSerialization:
    """Tests for Transcript.to_dict() / Transcript.from_dict()."""

    def test_empty_transcript_roundtrip(self) -> None:
        """Minimal transcript survives round-trip."""
        t = Transcript(task_id="task-1")
        data = t.to_dict()
        restored = Transcript.from_dict(data)

        assert restored.transcript_id == t.transcript_id
        assert restored.task_id == "task-1"
        assert restored.steps == []
        assert restored.final_output is None

    def test_full_transcript_roundtrip(self, sample_transcript: Transcript) -> None:
        """Full transcript with steps, tool calls, timing survives round-trip."""
        data = sample_transcript.to_dict()
        restored = Transcript.from_dict(data)

        assert restored.transcript_id == sample_transcript.transcript_id
        assert restored.task_id == sample_transcript.task_id
        assert restored.agent_name == sample_transcript.agent_name
        assert restored.agent_version == sample_transcript.agent_version
        assert len(restored.steps) == len(sample_transcript.steps)
        assert restored.final_output == sample_transcript.final_output
        assert len(restored.tool_calls) == len(sample_transcript.tool_calls)
        assert len(restored.errors) == len(sample_transcript.errors)

    def test_transcript_timestamps_roundtrip(self) -> None:
        """Datetime fields serialize to ISO strings and back."""
        now = datetime.now(timezone.utc)
        t = Transcript(task_id="task-1", started_at=now, completed_at=now)
        data = t.to_dict()

        # Timestamps should be ISO strings in the dict
        assert isinstance(data["started_at"], str)

        restored = Transcript.from_dict(data)
        assert restored.started_at is not None
        assert restored.completed_at is not None

    def test_transcript_with_decision_spec_roundtrip(self) -> None:
        """Transcript with DecisionSpec survives round-trip."""
        spec = DecisionSpec(
            model=ModelConfig(provider="anthropic", model_id="claude-3-opus"),
            prompts=PromptSpec.from_prompts(system_prompt="You are helpful"),
            tools=[ToolSpec(name="search", version="1.0")],
            agent=AgentSpec(agent_name="test_agent", agent_version="1.0"),
            environment=EnvironmentSpec(git_commit="abc123"),
        )
        t = Transcript(task_id="task-1", decision_spec=spec)
        data = t.to_dict()
        restored = Transcript.from_dict(data)

        assert restored.decision_spec is not None
        assert restored.decision_spec.fingerprint == spec.fingerprint
        assert restored.decision_spec.model.provider == "anthropic"

    def test_transcript_step_types_preserved(self) -> None:
        """Step types (enums) are preserved through serialization."""
        t = Transcript(task_id="task-1")
        t.add_step(TranscriptStep(step_type=StepType.LLM_CALL, content="hello"))
        t.add_step(TranscriptStep(
            step_type=StepType.TOOL_CALL,
            tool_call=ToolCall(
                tool_name="search",
                arguments={"q": "test"},
                result=["result1"],
                duration_ms=100.0,
            ),
        ))
        t.add_step(TranscriptStep(step_type=StepType.ERROR, error="boom"))

        data = t.to_dict()
        restored = Transcript.from_dict(data)

        assert restored.steps[0].step_type == StepType.LLM_CALL
        assert restored.steps[1].step_type == StepType.TOOL_CALL
        assert restored.steps[1].tool_call is not None
        assert restored.steps[1].tool_call.tool_name == "search"
        assert restored.steps[2].step_type == StepType.ERROR
        assert restored.steps[2].error == "boom"

    def test_transcript_to_dict_is_json_serializable(
        self, sample_transcript: Transcript
    ) -> None:
        """to_dict() output can be passed to json.dumps without errors."""
        data = sample_transcript.to_dict()
        json_str = json.dumps(data, default=str)
        assert isinstance(json_str, str)


class TestTrialSerialization:
    """Tests for Trial round-trip via TrialBatch."""

    def test_trial_roundtrip(self, sample_trial: Trial) -> None:
        """Single trial survives round-trip via TrialBatch."""
        batch = TrialBatch(trials=[sample_trial])
        data = batch.to_dict()
        restored = TrialBatch.from_dict(data)

        assert len(restored.trials) == 1
        trial = restored.trials[0]
        assert trial.trial_id == sample_trial.trial_id
        assert trial.task_id == sample_trial.task_id
        assert trial.status == TrialStatus.COMPLETED
        assert trial.run_index == sample_trial.run_index
        assert trial.total_runs == sample_trial.total_runs

    def test_trial_outcomes_roundtrip(self, sample_trial: Trial) -> None:
        """Trial outcomes survive round-trip."""
        batch = TrialBatch(trials=[sample_trial])
        data = batch.to_dict()
        restored = TrialBatch.from_dict(data)

        trial = restored.trials[0]
        assert len(trial.outcomes) == len(sample_trial.outcomes)
        assert trial.outcomes[0].grader_id == "quality_grader"
        assert trial.outcomes[0].passed is True
        assert trial.outcomes[0].score == pytest.approx(0.85)
        assert trial.outcomes[0].metrics == sample_trial.outcomes[0].metrics


class TestTrialBatchSerialization:
    """Tests for TrialBatch.to_dict() / TrialBatch.from_dict()."""

    def test_empty_batch_roundtrip(self) -> None:
        """Empty batch survives round-trip."""
        batch = TrialBatch()
        data = batch.to_dict()
        restored = TrialBatch.from_dict(data)

        assert restored.batch_id == batch.batch_id
        assert restored.trials == []

    def test_batch_with_multiple_trials_roundtrip(self) -> None:
        """Batch with several trials preserves all data."""
        trials = []
        for i in range(3):
            trial = Trial(
                task_id=f"task-{i}",
                run_index=i,
                total_runs=3,
                status=TrialStatus.COMPLETED,
            )
            trial.add_outcome(Outcome(
                trial_id=trial.trial_id,
                grader_id="g1",
                passed=i % 2 == 0,
                score=0.5 + i * 0.1,
            ))
            trials.append(trial)

        batch = TrialBatch(
            trials=trials,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        data = batch.to_dict()
        restored = TrialBatch.from_dict(data)

        assert restored.batch_id == batch.batch_id
        assert len(restored.trials) == 3
        assert restored.started_at is not None
        assert restored.completed_at is not None

        # Verify pass results preserved
        assert restored.trials[0].passed is True
        assert restored.trials[1].passed is False
        assert restored.trials[2].passed is True

    def test_batch_to_dict_is_json_serializable(self) -> None:
        """to_dict() output can be passed to json.dumps."""
        trial = Trial(
            task_id="t1",
            status=TrialStatus.COMPLETED,
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id,
            grader_id="g1",
            passed=True,
            score=0.9,
        ))
        batch = TrialBatch(
            trials=[trial],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        data = batch.to_dict()
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Verify round-trip through JSON
        parsed = json.loads(json_str)
        restored = TrialBatch.from_dict(parsed)
        assert len(restored.trials) == 1

    def test_batch_with_failed_trials(self) -> None:
        """Batch with error info preserved."""
        trial = Trial(
            task_id="t1",
            status=TrialStatus.FAILED,
            error_message="Connection refused",
            error_traceback="Traceback...",
        )
        batch = TrialBatch(trials=[trial])
        data = batch.to_dict()
        restored = TrialBatch.from_dict(data)

        assert restored.trials[0].status == TrialStatus.FAILED
        assert restored.trials[0].error_message == "Connection refused"
        assert restored.trials[0].error_traceback == "Traceback..."

    def test_batch_with_transcript_roundtrip(
        self, sample_trial: Trial
    ) -> None:
        """Batch with full transcript data survives round-trip."""
        batch = TrialBatch(trials=[sample_trial])
        data = batch.to_dict()
        restored = TrialBatch.from_dict(data)

        transcript = restored.trials[0].transcript
        assert transcript is not None
        assert transcript.task_id == sample_trial.transcript.task_id
        assert transcript.agent_name == sample_trial.transcript.agent_name
        assert len(transcript.steps) == len(sample_trial.transcript.steps)
        assert transcript.final_output == sample_trial.transcript.final_output
