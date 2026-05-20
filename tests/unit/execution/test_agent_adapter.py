"""Tests for agent adapter module."""

from datetime import UTC, datetime

import pytest

from tracelens.core.task import Task
from tracelens.core.transcript import StepType, Transcript
from tracelens.execution.agent_adapter import AgentAdapter, SimpleAdapter


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


class _LifecycleTrackingAdapter(AgentAdapter):
    """Records call order for lifecycle hook testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.setup_error: Exception | None = None
        self.run_error: Exception | None = None
        self.teardown_error: Exception | None = None

    async def setup(self, task: Task) -> None:
        self.calls.append("setup")
        if self.setup_error:
            raise self.setup_error

    async def run(self, task: Task) -> Transcript:
        self.calls.append("run")
        if self.run_error:
            raise self.run_error
        transcript = self.start_transcript(task)
        transcript.final_output = "ok"
        transcript.completed_at = datetime.now(UTC)
        return transcript

    async def teardown(self, task: Task, transcript: Transcript | None = None) -> None:
        self.calls.append("teardown")
        if self.teardown_error:
            raise self.teardown_error


class TestAgentAdapterLifecycleHooks:
    """Tests for setup/teardown lifecycle hooks."""

    @pytest.fixture
    def task(self) -> Task:
        return Task(task_id="t1", name="Test", input_data={"goal": "test"})

    async def test_default_hooks_are_noop(self, task: Task):
        """Default setup/teardown do nothing and don't raise."""
        async def fn(data: dict) -> str:
            return "ok"

        adapter = SimpleAdapter(fn)
        # Should not raise
        await adapter.setup(task)
        await adapter.teardown(task, None)

    async def test_setup_called_before_run(self, task: Task):
        """Lifecycle order: setup -> run -> teardown."""
        adapter = _LifecycleTrackingAdapter()
        await adapter.setup(task)
        transcript = await adapter.run(task)
        await adapter.teardown(task, transcript)

        assert adapter.calls == ["setup", "run", "teardown"]

    async def test_teardown_receives_none_on_setup_failure(self, task: Task):
        """When setup fails, teardown is called with transcript=None."""
        adapter = _LifecycleTrackingAdapter()
        adapter.setup_error = RuntimeError("setup boom")

        with pytest.raises(RuntimeError, match="setup boom"):
            await adapter.setup(task)

        # Teardown still callable with None
        await adapter.teardown(task, None)
        assert "teardown" in adapter.calls


class TestAgentAdapterABC:
    """Tests for the AgentAdapter abstract base class."""

    def test_cannot_instantiate_directly(self):
        """AgentAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            AgentAdapter()
