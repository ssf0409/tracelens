"""Transcript models for recording agent execution.

A Transcript is a complete record of an agent's execution, including:
- All steps taken (LLM calls, tool calls, etc.)
- Final and intermediate outputs
- Timing and token usage
- Errors encountered
- Decision specification (for reproducibility)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from eval_kit.core.decision_spec import DecisionSpec


class StepType(StrEnum):
    """Types of execution steps."""

    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    USER_INPUT = "user_input"
    AGENT_OUTPUT = "agent_output"
    ERROR = "error"
    INTERNAL = "internal"


class StreamingEventType(StrEnum):
    """Types of streaming events for real-time token delivery."""

    STREAM_START = "stream_start"
    TOKEN = "token"
    CHUNK = "chunk"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    STREAM_END = "stream_end"
    ERROR = "error"


class StreamingEvent(BaseModel):
    """A single event in a streaming response.

    Timestamps are in milliseconds since stream start for precise
    inter-token latency analysis.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: StreamingEventType
    timestamp_ms: float
    content: str | None = None
    token_count: int | None = None
    sse_event: str | None = None
    sse_data: str | None = None
    tool_name: str | None = None


class ToolCall(BaseModel):
    """Record of a tool invocation.

    Captures the tool name, arguments, result, and timing.
    """

    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    error: str | None = None
    duration_ms: float | None = None


class TranscriptStep(BaseModel):
    """A single step in agent execution.

    Steps are the atomic units of a transcript. Each step represents
    one action the agent took (LLM call, tool call, etc.).
    """

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_type: StepType
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Content depends on step type
    content: Any = None
    tool_call: ToolCall | None = None

    # LLM-specific fields
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    # Error information
    error: str | None = None
    error_traceback: str | None = None

    @property
    def is_error(self) -> bool:
        """Check if this step is an error."""
        return self.step_type == StepType.ERROR or self.error is not None


class Transcript(BaseModel):
    """Complete record of agent execution for a task.

    A Transcript captures everything that happened during an agent's
    execution, providing a complete audit trail for grading and debugging.

    Example:
        transcript = Transcript(
            transcript_id=str(uuid.uuid4()),
            task_id=task.task_id,
            agent_name="goal_decomposition",
            started_at=datetime.utcnow(),
        )
        transcript.steps.append(step)
        transcript.final_output = result
        transcript.completed_at = datetime.utcnow()
    """

    transcript_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str

    # Execution timeline
    steps: list[TranscriptStep] = Field(default_factory=list)

    # Summary outputs
    final_output: Any = None
    intermediate_outputs: list[Any] = Field(default_factory=list)

    # Aggregated tool calls for easy access
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Errors encountered
    errors: list[str] = Field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Metadata
    agent_name: str | None = None
    agent_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Reproducibility
    decision_spec: DecisionSpec | None = Field(
        default=None,
        description="DecisionSpec capturing all parameters affecting agent behavior",
    )

    # Streaming data
    streaming_events: list[StreamingEvent] = Field(default_factory=list)

    @property
    def has_streaming_data(self) -> bool:
        """Whether this transcript contains streaming events."""
        return len(self.streaming_events) > 0

    @property
    def first_token_latency_ms(self) -> float | None:
        """Time to first token event in milliseconds, or None if no streaming."""
        for event in self.streaming_events:
            if event.event_type == StreamingEventType.TOKEN:
                return event.timestamp_ms
        return None

    @property
    def streaming_duration_ms(self) -> float | None:
        """Total streaming duration from first to last event, or None if no streaming."""
        if not self.streaming_events:
            return None
        return self.streaming_events[-1].timestamp_ms - self.streaming_events[0].timestamp_ms

    @property
    def streaming_token_count(self) -> int:
        """Total tokens across all streaming TOKEN events."""
        return sum(
            e.token_count or 0
            for e in self.streaming_events
            if e.event_type == StreamingEventType.TOKEN
        )

    def add_streaming_event(self, event: StreamingEvent) -> None:
        """Append a streaming event to the transcript."""
        self.streaming_events.append(event)

    @property
    def duration_ms(self) -> float | None:
        """Calculate execution duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def total_tokens(self) -> int:
        """Calculate total token usage across all steps."""
        return sum(
            (s.tokens_in or 0) + (s.tokens_out or 0)
            for s in self.steps
        )

    @property
    def input_tokens(self) -> int:
        """Calculate total input tokens."""
        return sum(s.tokens_in or 0 for s in self.steps)

    @property
    def output_tokens(self) -> int:
        """Calculate total output tokens."""
        return sum(s.tokens_out or 0 for s in self.steps)

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during execution."""
        return len(self.errors) > 0 or any(s.is_error for s in self.steps)

    @property
    def llm_calls_count(self) -> int:
        """Count LLM calls."""
        return sum(1 for s in self.steps if s.step_type == StepType.LLM_CALL)

    @property
    def tool_calls_count(self) -> int:
        """Count tool calls."""
        return sum(1 for s in self.steps if s.step_type == StepType.TOOL_CALL)

    def add_step(self, step: TranscriptStep) -> None:
        """Add a step to the transcript."""
        self.steps.append(step)
        if step.tool_call:
            self.tool_calls.append(step.tool_call)
        if step.error:
            self.errors.append(step.error)

    def get_tool_calls_by_name(self, name: str) -> list[ToolCall]:
        """Get all tool calls with a specific name."""
        return [tc for tc in self.tool_calls if tc.tool_name == name]

    def get_steps_by_type(self, step_type: StepType) -> list[TranscriptStep]:
        """Get all steps of a specific type."""
        return [s for s in self.steps if s.step_type == step_type]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict with full round-trip fidelity."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        """Reconstruct a Transcript from a dict produced by to_dict()."""
        return cls.model_validate(data)

    def to_summary(self) -> dict[str, Any]:
        """Create a summary dict for reporting."""
        summary = {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "duration_ms": self.duration_ms,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls_count,
            "tool_calls": self.tool_calls_count,
            "has_errors": self.has_errors,
            "error_count": len(self.errors),
        }
        if self.decision_spec:
            summary["fingerprint"] = self.decision_spec.fingerprint_short
        return summary


# Rebuild model to resolve forward reference to DecisionSpec
def _rebuild_transcript_model() -> None:
    """Rebuild Transcript model with DecisionSpec reference resolved."""
    from eval_kit.core.decision_spec import DecisionSpec  # noqa: PLC0415

    Transcript.model_rebuild(_types_namespace={"DecisionSpec": DecisionSpec})


_rebuild_transcript_model()
