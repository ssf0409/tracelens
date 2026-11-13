"""Agent adapter interface for invoking agents during evaluation.

An AgentAdapter wraps an agent so the evaluation runner can invoke it
on tasks and collect transcripts.
"""

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from tracelens.core._time import utc_now
from tracelens.core.task import Task
from tracelens.core.transcript import StepType, Transcript, TranscriptStep


class AgentAdapter(ABC):
    """Abstract base class for agent adapters.

    Adapters bridge the evaluation runner to the agent being evaluated.
    Implement `run()` to invoke your agent and return a Transcript.

    Optionally override `setup()` and `teardown()` for lifecycle management.
    The runner guarantees teardown is called even if run() fails.

    Asyncio Contract & Blocking Operations:
        TraceLens runs on an asyncio event loop. Adapter hooks (``setup()``,
        ``run()``, ``teardown()``) must not execute blocking synchronous calls
        (e.g., ``time.sleep()``, synchronous HTTP clients, heavy CPU computation)
        directly on the event loop thread, as this prevents cooperative timeouts
        (``asyncio.wait_for``) and task cancellation from executing promptly.

        If your agent or hooks perform synchronous blocking operations, offload
        them to a worker thread via ``asyncio.to_thread``, or use ``SyncAdapter`` /
        pass the sync callable to ``SimpleAdapter``.

    Example:
        class MyAdapter(AgentAdapter):
            async def setup(self, task: Task) -> None:
                self.db = await create_test_database()

            async def run(self, task: Task) -> Transcript:
                result = await my_agent.invoke(task.input_data)
                transcript = self.start_transcript(task)
                transcript.final_output = result
                transcript.completed_at = utc_now()
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
            started_at=utc_now(),
        )

    def record_error(self, transcript: Transcript, error: Exception) -> None:
        """Helper to record an exception in a transcript."""
        transcript.errors.append(str(error))
        transcript.add_step(
            TranscriptStep(
                step_type=StepType.ERROR,
                error=str(error),
            )
        )
        transcript.completed_at = utc_now()


class SimpleAdapter(AgentAdapter):
    """Wraps an async or sync callable as an AgentAdapter.

    Useful for testing and simple single-shot agents that take
    input_data and return a result. Synchronous callables are
    automatically offloaded to a worker thread using ``asyncio.to_thread``
    so they do not block the event loop and allow runner timeouts to trigger.

    Example:
        async def my_fn(input_data: dict) -> dict:
            return {"answer": "42"}

        adapter = SimpleAdapter(my_fn)
    """

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], Awaitable[Any] | Any],
    ) -> None:
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)

    async def run(self, task: Task) -> Transcript:
        """Invoke the wrapped function and build a transcript."""
        transcript = self.start_transcript(task)
        try:
            if self._is_async:
                result = await self._fn(task.input_data)
            else:
                result = await asyncio.to_thread(self._fn, task.input_data)
            transcript.final_output = result
            transcript.add_step(
                TranscriptStep(
                    step_type=StepType.AGENT_OUTPUT,
                    content=result,
                )
            )
        except Exception as exc:
            self.record_error(transcript, exc)
            raise
        finally:
            transcript.completed_at = utc_now()
        return transcript


class SyncAdapter(AgentAdapter):
    """Wraps a synchronous callable as an AgentAdapter offloaded to a thread worker.

    Ensures that synchronous agents (or agents using synchronous libraries like
    `time.sleep()`, synchronous `requests`, etc.) do not block the asyncio event
    loop, allowing the runner's per-trial timeouts to trigger cooperatively.

    Example:
        def my_sync_agent(input_data: dict) -> dict:
            return {"answer": "42"}

        adapter = SyncAdapter(my_sync_agent)
    """

    def __init__(self, fn: Callable[[dict[str, Any]], Any]) -> None:
        self._fn = fn

    async def run(self, task: Task) -> Transcript:
        """Invoke the synchronous callable in a thread worker and build a transcript."""
        transcript = self.start_transcript(task)
        try:
            result = await asyncio.to_thread(self._fn, task.input_data)
            transcript.final_output = result
            transcript.add_step(
                TranscriptStep(
                    step_type=StepType.AGENT_OUTPUT,
                    content=result,
                )
            )
        except Exception as exc:
            self.record_error(transcript, exc)
            raise
        finally:
            transcript.completed_at = utc_now()
        return transcript
