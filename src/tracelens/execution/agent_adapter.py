"""Agent adapter interface for invoking agents during evaluation.

An AgentAdapter wraps an agent so the evaluation runner can invoke it
on tasks and collect transcripts.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from tracelens.core._time import utc_now
from tracelens.core.task import Task
from tracelens.core.transcript import StepType, ToolCall, Transcript, TranscriptStep


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
        transcript.add_step(TranscriptStep(
            step_type=StepType.ERROR,
            error=str(error),
        ))
        transcript.completed_at = utc_now()

    def record_llm_call(
        self,
        transcript: Transcript,
        *,
        model: str | None = None,
        content: Any = None,
        tokens_in: int | None = None,
        tokens_out: int | None = None,
    ) -> TranscriptStep:
        """Helper to record an LLM call with token counts."""
        step = TranscriptStep(
            step_type=StepType.LLM_CALL,
            model=model,
            content=content,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
        transcript.add_step(step)
        return step

    def record_tool_call(
        self,
        transcript: Transcript,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any | None = None,
        error: str | None = None,
        duration_ms: float | None = None,
    ) -> TranscriptStep:
        """Helper to record a tool call."""
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )
        step = TranscriptStep(
            step_type=StepType.TOOL_CALL,
            tool_call=tool_call,
        )
        transcript.add_step(step)
        return step


class SimpleAdapter(AgentAdapter):
    """Wraps any async callable as an AgentAdapter.

    Useful for testing and simple single-shot agents that take
    input_data and return a result. Optionally accepts a usage extractor
    returning (tokens_in, tokens_out).

    Example:
        async def my_fn(input_data: dict) -> dict:
            return {"answer": "42"}

        adapter = SimpleAdapter(my_fn)
    """

    def __init__(
        self,
        fn: Callable[[dict[str, Any]], Awaitable[Any]],
        *,
        usage_fn: Callable[[Any], tuple[int, int]] | None = None,
    ) -> None:
        self._fn = fn
        self._usage_fn = usage_fn

    async def run(self, task: Task) -> Transcript:
        """Invoke the wrapped function and build a transcript."""
        transcript = self.start_transcript(task)
        try:
            result = await self._fn(task.input_data)
            transcript.final_output = result
            tokens_in, tokens_out = (None, None)
            if self._usage_fn is not None:
                try:
                    tokens_in, tokens_out = self._usage_fn(result)
                except Exception:
                    pass
            transcript.add_step(TranscriptStep(
                step_type=StepType.AGENT_OUTPUT,
                content=result,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            ))
        except Exception as exc:
            self.record_error(transcript, exc)
            raise
        finally:
            transcript.completed_at = utc_now()
        return transcript
