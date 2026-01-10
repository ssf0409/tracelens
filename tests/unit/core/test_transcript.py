"""Tests for transcript module."""

import pytest
from datetime import datetime, timedelta

from agent_eval.core.transcript import (
    Transcript,
    TranscriptStep,
    StepType,
    ToolCall,
)


class TestToolCall:
    """Tests for ToolCall model."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tc = ToolCall(
            tool_name="search",
            arguments={"query": "test"},
            result=["result1", "result2"],
            duration_ms=150.0,
        )

        assert tc.tool_name == "search"
        assert tc.arguments == {"query": "test"}
        assert tc.result == ["result1", "result2"]
        assert tc.duration_ms == 150.0
        assert tc.error is None

    def test_tool_call_with_error(self):
        """Test tool call with error."""
        tc = ToolCall(
            tool_name="search",
            arguments={"query": "test"},
            error="Connection timeout",
        )

        assert tc.error == "Connection timeout"
        assert tc.result is None


class TestTranscriptStep:
    """Tests for TranscriptStep model."""

    def test_step_creation(self):
        """Test creating a transcript step."""
        step = TranscriptStep(
            step_id="step-1",
            step_type=StepType.LLM_CALL,
            content="Processing...",
            model="gpt-4",
            tokens_in=100,
            tokens_out=50,
        )

        assert step.step_id == "step-1"
        assert step.step_type == StepType.LLM_CALL
        assert step.tokens_in == 100
        assert step.tokens_out == 50
        assert not step.is_error

    def test_step_with_error(self):
        """Test step with error."""
        step = TranscriptStep(
            step_id="step-1",
            step_type=StepType.ERROR,
            error="Something went wrong",
        )

        assert step.is_error
        assert step.error == "Something went wrong"

    def test_step_with_tool_call(self):
        """Test step with tool call."""
        step = TranscriptStep(
            step_id="step-1",
            step_type=StepType.TOOL_CALL,
            tool_call=ToolCall(
                tool_name="calculate",
                arguments={"x": 1, "y": 2},
                result=3,
            ),
        )

        assert step.tool_call is not None
        assert step.tool_call.tool_name == "calculate"


class TestTranscript:
    """Tests for Transcript model."""

    def test_transcript_creation(self, sample_task):
        """Test creating a transcript."""
        transcript = Transcript(
            task_id=sample_task.task_id,
            agent_name="test_agent",
        )

        assert transcript.task_id == sample_task.task_id
        assert transcript.agent_name == "test_agent"
        assert transcript.steps == []
        assert transcript.errors == []

    def test_transcript_add_step(self, sample_task):
        """Test adding steps to transcript."""
        transcript = Transcript(task_id=sample_task.task_id)

        step = TranscriptStep(
            step_id="step-1",
            step_type=StepType.LLM_CALL,
            tokens_in=100,
            tokens_out=50,
        )
        transcript.add_step(step)

        assert len(transcript.steps) == 1
        assert transcript.total_tokens == 150

    def test_transcript_add_step_with_tool_call(self, sample_task):
        """Test adding step with tool call."""
        transcript = Transcript(task_id=sample_task.task_id)

        step = TranscriptStep(
            step_id="step-1",
            step_type=StepType.TOOL_CALL,
            tool_call=ToolCall(tool_name="search", arguments={}),
        )
        transcript.add_step(step)

        assert len(transcript.tool_calls) == 1
        assert transcript.tool_calls[0].tool_name == "search"

    def test_transcript_add_step_with_error(self, sample_task):
        """Test adding step with error."""
        transcript = Transcript(task_id=sample_task.task_id)

        step = TranscriptStep(
            step_id="step-1",
            step_type=StepType.ERROR,
            error="Something failed",
        )
        transcript.add_step(step)

        assert len(transcript.errors) == 1
        assert transcript.has_errors

    def test_transcript_duration(self, sample_transcript):
        """Test duration calculation."""
        duration = sample_transcript.duration_ms

        assert duration is not None
        assert duration >= 0

    def test_transcript_token_counts(self, sample_transcript):
        """Test token counting."""
        assert sample_transcript.total_tokens == 150  # 100 + 50 from first step
        assert sample_transcript.input_tokens == 100
        assert sample_transcript.output_tokens == 50

    def test_transcript_llm_calls_count(self, sample_transcript):
        """Test LLM calls counting."""
        assert sample_transcript.llm_calls_count == 1

    def test_transcript_tool_calls_count(self, sample_transcript):
        """Test tool calls counting."""
        assert sample_transcript.tool_calls_count == 1

    def test_transcript_get_tool_calls_by_name(self, sample_transcript):
        """Test getting tool calls by name."""
        search_calls = sample_transcript.get_tool_calls_by_name("search")
        assert len(search_calls) == 1

        other_calls = sample_transcript.get_tool_calls_by_name("other")
        assert len(other_calls) == 0

    def test_transcript_get_steps_by_type(self, sample_transcript):
        """Test getting steps by type."""
        llm_steps = sample_transcript.get_steps_by_type(StepType.LLM_CALL)
        assert len(llm_steps) == 1

        tool_steps = sample_transcript.get_steps_by_type(StepType.TOOL_CALL)
        assert len(tool_steps) == 1

    def test_transcript_to_summary(self, sample_transcript):
        """Test transcript summary generation."""
        summary = sample_transcript.to_summary()

        assert summary["task_id"] == sample_transcript.task_id
        assert summary["agent_name"] == "test_agent"
        assert summary["llm_calls"] == 1
        assert summary["tool_calls"] == 1
        assert summary["has_errors"] is False
