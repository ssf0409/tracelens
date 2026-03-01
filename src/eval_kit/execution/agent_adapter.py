"""Agent adapter interface for invoking agents during evaluation.

An AgentAdapter wraps an agent so the evaluation runner can invoke it
on tasks and collect transcripts.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from eval_kit.core.task import Task
from eval_kit.core.transcript import StepType, Transcript, TranscriptStep


class AgentAdapter(ABC):
    """Abstract base class for agent adapters.

    Adapters bridge the evaluation runner to the agent being evaluated.
    Implement `run()` to invoke your agent and return a Transcript.

    Optionally override `setup()` and `teardown()` for lifecycle management.
    The runner guarantees teardown is called even if run() fails.

    Example:
        class MyAdapter(AgentAdapter):
            async def setup(self, task: Task) -> None:
                self.db = await create_test_database()

            async def run(self, task: Task) -> Transcript:
                result = await my_agent.invoke(task.input_data)
                transcript = self.start_transcript(task)
                transcript.final_output = result
                transcript.completed_at = datetime.utcnow()
                return transcript

            async def teardown(self, task: Task, transcript: Transcript | None) -> None:
                await self.db.cleanup()
    """

    async def setup(self, task: Task) -> None:
        """Called before run(). Override for preparation. Default: no-op."""

    async def teardown(self, task: Task, transcript: Transcript | None) -> None:
        """Called after run(), even on failure. Override for cleanup. Default: no-op."""

    @abstractmethod
    async def run(self, task: Task) -> Transcript:
        """Run the agent on a task and return a transcript."""
        ...

    def start_transcript(self, task: Task) -> Transcript:
        """Helper to create a Transcript with timing started."""
        return Transcript(
            task_id=task.task_id,
            started_at=datetime.utcnow(),
        )

    def record_error(self, transcript: Transcript, error: Exception) -> None:
        """Helper to record an exception in a transcript."""
        transcript.errors.append(str(error))
        transcript.add_step(TranscriptStep(
            step_type=StepType.ERROR,
            error=str(error),
        ))
        transcript.completed_at = datetime.utcnow()


class SimpleAdapter(AgentAdapter):
    """Wraps any async callable as an AgentAdapter.

    Useful for testing and simple single-shot agents that take
    input_data and return a result.

    Example:
        async def my_fn(input_data: dict) -> dict:
            return {"answer": "42"}

        adapter = SimpleAdapter(my_fn)
    """

    def __init__(self, fn: Callable[[dict[str, Any]], Awaitable[Any]]) -> None:
        self._fn = fn

    async def run(self, task: Task) -> Transcript:
        """Invoke the wrapped function and build a transcript."""
        transcript = self.start_transcript(task)
        try:
            result = await self._fn(task.input_data)
            transcript.final_output = result
            transcript.add_step(TranscriptStep(
                step_type=StepType.AGENT_OUTPUT,
                content=result,
            ))
        except Exception as exc:
            self.record_error(transcript, exc)
            raise
        finally:
            transcript.completed_at = datetime.utcnow()
        return transcript
