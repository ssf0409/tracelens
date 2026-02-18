"""Tests for agent adapter module."""

import pytest

from eval_kit.core.task import Task
from eval_kit.core.transcript import StepType
from eval_kit.execution.agent_adapter import AgentAdapter, SimpleAdapter


class TestSimpleAdapter:
    """Tests for SimpleAdapter wrapping an async function."""

    @pytest.fixture
    def task(self) -> Task:
        return Task(task_id="t1", name="Test", input_data={"goal": "test"})

    async def test_successful_run(self, task: Task):
        """SimpleAdapter returns a transcript with the function's output."""
        async def fn(input_data: dict) -> dict:
            return {"answer": input_data["goal"]}

        adapter = SimpleAdapter(fn)
        transcript = await adapter.run(task)

        assert transcript.task_id == "t1"
        assert transcript.final_output == {"answer": "test"}
        assert transcript.started_at is not None
        assert transcript.completed_at is not None
        assert not transcript.has_errors

    async def test_records_agent_output_step(self, task: Task):
        """SimpleAdapter adds an AGENT_OUTPUT step."""
        async def fn(input_data: dict) -> str:
            return "result"

        adapter = SimpleAdapter(fn)
        transcript = await adapter.run(task)

        assert len(transcript.steps) == 1
        assert transcript.steps[0].step_type == StepType.AGENT_OUTPUT
        assert transcript.steps[0].content == "result"

    async def test_error_recording(self, task: Task):
        """SimpleAdapter records errors and re-raises."""
        async def fn(input_data: dict) -> dict:
            raise ValueError("boom")

        adapter = SimpleAdapter(fn)
        with pytest.raises(ValueError, match="boom"):
            await adapter.run(task)

    async def test_start_transcript_helper(self, task: Task):
        """start_transcript creates a Transcript with correct task_id and timing."""
        async def fn(input_data: dict) -> str:
            return "ok"

        adapter = SimpleAdapter(fn)
        transcript = adapter.start_transcript(task)

        assert transcript.task_id == task.task_id
        assert transcript.started_at is not None

    async def test_record_error_helper(self, task: Task):
        """record_error adds error info to the transcript."""
        async def fn(input_data: dict) -> str:
            return "ok"

        adapter = SimpleAdapter(fn)
        transcript = adapter.start_transcript(task)
        adapter.record_error(transcript, ValueError("test error"))

        assert transcript.has_errors
        assert "test error" in transcript.errors[0]
        assert transcript.completed_at is not None


class TestAgentAdapterABC:
    """Tests for the AgentAdapter abstract base class."""

    def test_cannot_instantiate_directly(self):
        """AgentAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            AgentAdapter()
