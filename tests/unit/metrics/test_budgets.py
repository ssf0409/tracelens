"""Tests for budget/signal metric graders.

Tests LatencyGrader, TokenBudgetGrader, ToolCallGrader, and TraceConsistencyGrader.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from tracelens.core._time import utc_now
from tracelens.core.grader import EvalPolicy, GraderConfig
from tracelens.core.task import Task
from tracelens.core.transcript import (
    StepType,
    ToolCall,
    Transcript,
    TranscriptStep,
)
from tracelens.metrics.budgets import (
    LatencyGrader,
    TokenBudgetGrader,
    ToolCallGrader,
    TraceConsistencyGrader,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def task() -> Task:
    return Task(
        task_id="task-1",
        name="test task",
        input_data={"prompt": "hello"},
    )


def _make_transcript(
    *,
    duration_ms: float | None = None,
    steps: list[TranscriptStep] | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> Transcript:
    """Build a Transcript with optional timing and tool calls."""
    now = utc_now()
    started = now - timedelta(milliseconds=duration_ms or 0)
    completed = now if duration_ms is not None else None

    return Transcript(
        task_id="task-1",
        started_at=started if duration_ms is not None else None,
        completed_at=completed,
        steps=steps or [],
        tool_calls=tool_calls or [],
    )


# ---------------------------------------------------------------------------
# LatencyGrader
# ---------------------------------------------------------------------------

class TestLatencyGrader:

    def test_default_policy_is_warn(self) -> None:
        grader = LatencyGrader("latency", max_ms=1000.0)
        assert grader.policy == EvalPolicy.WARN

    def test_custom_policy(self) -> None:
        cfg = GraderConfig(policy=EvalPolicy.GATE)
        grader = LatencyGrader("latency", max_ms=1000.0, config=cfg)
        assert grader.policy == EvalPolicy.GATE

    def test_zero_max_ms_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ms must be positive"):
            LatencyGrader("latency", max_ms=0.0)

    def test_negative_max_ms_raises(self) -> None:
        with pytest.raises(ValueError, match="max_ms must be positive"):
            LatencyGrader("latency", max_ms=-100.0)

    def test_pass_under_budget(self, task: Task) -> None:
        grader = LatencyGrader("latency", max_ms=1000.0)
        transcript = _make_transcript(duration_ms=500.0)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["duration_ms"] == pytest.approx(500.0, abs=50)
        assert metrics["budget_ratio"] == pytest.approx(0.5, abs=0.05)

        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == pytest.approx(0.5, abs=0.05)

    def test_fail_over_budget(self, task: Task) -> None:
        grader = LatencyGrader("latency", max_ms=1000.0)
        transcript = _make_transcript(duration_ms=1500.0)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["duration_ms"] == pytest.approx(1500.0, abs=50)
        assert metrics["budget_ratio"] == pytest.approx(1.5, abs=0.05)

        passed, score = grader.determine_pass(metrics, task)
        assert passed is False
        assert score == 0.0

    def test_exact_budget_passes(self, task: Task) -> None:
        grader = LatencyGrader("latency", max_ms=1000.0)
        transcript = _make_transcript(duration_ms=1000.0)

        metrics = grader.compute_metrics(transcript, task)
        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == pytest.approx(0.0, abs=0.05)

    def test_no_duration_fails(self, task: Task) -> None:
        grader = LatencyGrader("latency", max_ms=1000.0)
        transcript = _make_transcript()  # no duration

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["duration_ms"] == 0.0
        assert metrics["budget_ratio"] == 0.0

    def test_grade_async(self, task: Task) -> None:
        grader = LatencyGrader("latency", max_ms=2000.0)
        transcript = _make_transcript(duration_ms=1000.0)

        outcome = asyncio.run(grader.grade(transcript, task))
        assert outcome.passed is True
        assert "duration_ms" in outcome.metrics
        assert outcome.score == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# TokenBudgetGrader
# ---------------------------------------------------------------------------

class TestTokenBudgetGrader:

    def test_default_policy_is_warn(self) -> None:
        grader = TokenBudgetGrader("tokens", max_tokens=1000)
        assert grader.policy == EvalPolicy.WARN

    def test_zero_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            TokenBudgetGrader("tokens", max_tokens=0)

    def test_negative_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            TokenBudgetGrader("tokens", max_tokens=-10)

    def test_pass_under_budget(self, task: Task) -> None:
        grader = TokenBudgetGrader("tokens", max_tokens=1000)
        steps = [
            TranscriptStep(
                step_type=StepType.LLM_CALL,
                tokens_in=200,
                tokens_out=100,
            ),
        ]
        transcript = _make_transcript(steps=steps)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["total_tokens"] == 300.0
        assert metrics["budget_ratio"] == pytest.approx(0.3)

        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == pytest.approx(0.7)

    def test_fail_over_budget(self, task: Task) -> None:
        grader = TokenBudgetGrader("tokens", max_tokens=500)
        steps = [
            TranscriptStep(
                step_type=StepType.LLM_CALL,
                tokens_in=400,
                tokens_out=300,
            ),
        ]
        transcript = _make_transcript(steps=steps)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["total_tokens"] == 700.0
        assert metrics["budget_ratio"] == pytest.approx(1.4)

        passed, score = grader.determine_pass(metrics, task)
        assert passed is False
        assert score == 0.0

    def test_zero_tokens_passes(self, task: Task) -> None:
        grader = TokenBudgetGrader("tokens", max_tokens=1000)
        transcript = _make_transcript()

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["total_tokens"] == 0.0

        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == 1.0

    def test_grade_async(self, task: Task) -> None:
        grader = TokenBudgetGrader("tokens", max_tokens=1000)
        steps = [
            TranscriptStep(step_type=StepType.LLM_CALL, tokens_in=100, tokens_out=100),
        ]
        transcript = _make_transcript(steps=steps)

        outcome = asyncio.run(grader.grade(transcript, task))
        assert outcome.passed is True
        assert outcome.metrics["total_tokens"] == 200.0


# ---------------------------------------------------------------------------
# ToolCallGrader
# ---------------------------------------------------------------------------

class TestToolCallGrader:

    def test_default_policy_is_gate(self) -> None:
        grader = ToolCallGrader("tools", required_tools=["search"])
        assert grader.policy == EvalPolicy.GATE

    def test_required_tools_all_called(self, task: Task) -> None:
        grader = ToolCallGrader("tools", required_tools=["search", "fetch"])
        tool_calls = [
            ToolCall(tool_name="search", arguments={}),
            ToolCall(tool_name="fetch", arguments={}),
        ]
        transcript = _make_transcript(tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["required_called"] == 1.0

        passed, _ = grader.determine_pass(metrics, task)
        assert passed is True

    def test_required_tools_missing(self, task: Task) -> None:
        grader = ToolCallGrader("tools", required_tools=["search", "fetch"])
        tool_calls = [
            ToolCall(tool_name="search", arguments={}),
        ]
        transcript = _make_transcript(tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["required_called"] == pytest.approx(0.5)

        passed, _ = grader.determine_pass(metrics, task)
        assert passed is False

    def test_allowed_tools_enforced(self, task: Task) -> None:
        grader = ToolCallGrader(
            "tools",
            allowed_tools=["search", "fetch"],
        )
        tool_calls = [
            ToolCall(tool_name="search", arguments={}),
            ToolCall(tool_name="delete", arguments={}),
        ]
        transcript = _make_transcript(tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["unauthorized_calls"] == 1.0

        passed, _ = grader.determine_pass(metrics, task)
        assert passed is False

    def test_forbidden_tools_enforced(self, task: Task) -> None:
        grader = ToolCallGrader("tools", forbidden_tools=["delete", "drop"])
        tool_calls = [
            ToolCall(tool_name="search", arguments={}),
            ToolCall(tool_name="delete", arguments={}),
        ]
        transcript = _make_transcript(tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["forbidden_calls"] == 1.0

        passed, _ = grader.determine_pass(metrics, task)
        assert passed is False

    def test_all_constraints_satisfied(self, task: Task) -> None:
        grader = ToolCallGrader(
            "tools",
            required_tools=["search"],
            allowed_tools=["search", "fetch"],
            forbidden_tools=["delete"],
        )
        tool_calls = [
            ToolCall(tool_name="search", arguments={}),
            ToolCall(tool_name="fetch", arguments={}),
        ]
        transcript = _make_transcript(tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == 1.0

    def test_no_tools_no_requirements_passes(self, task: Task) -> None:
        grader = ToolCallGrader("tools")
        transcript = _make_transcript()

        metrics = grader.compute_metrics(transcript, task)
        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == 1.0

    def test_no_tools_with_requirements_fails(self, task: Task) -> None:
        grader = ToolCallGrader("tools", required_tools=["search"])
        transcript = _make_transcript()

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["required_called"] == 0.0

        passed, _ = grader.determine_pass(metrics, task)
        assert passed is False

    def test_grade_async(self, task: Task) -> None:
        grader = ToolCallGrader("tools", required_tools=["search"])
        tool_calls = [ToolCall(tool_name="search", arguments={})]
        transcript = _make_transcript(tool_calls=tool_calls)

        outcome = asyncio.run(grader.grade(transcript, task))
        assert outcome.passed is True


# ---------------------------------------------------------------------------
# TraceConsistencyGrader
# ---------------------------------------------------------------------------

class TestTraceConsistencyGrader:

    def test_default_policy_is_warn(self) -> None:
        grader = TraceConsistencyGrader("consistency")
        assert grader.policy == EvalPolicy.WARN

    def test_no_errors_passes(self, task: Task) -> None:
        grader = TraceConsistencyGrader("consistency")
        tool_calls = [
            ToolCall(tool_name="search", arguments={}, result="found it"),
        ]
        steps = [
            TranscriptStep(
                step_type=StepType.TOOL_CALL,
                tool_call=tool_calls[0],
            ),
            TranscriptStep(step_type=StepType.AGENT_OUTPUT, content="answer"),
        ]
        transcript = _make_transcript(steps=steps, tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["tool_error_rate"] == 0.0
        assert metrics["phantom_calls"] == 0

        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == 1.0

    def test_high_error_rate_fails(self, task: Task) -> None:
        grader = TraceConsistencyGrader("consistency")
        tool_calls = [
            ToolCall(tool_name="search", arguments={}, error="timeout"),
            ToolCall(tool_name="fetch", arguments={}, error="404"),
        ]
        steps = [
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[0]),
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[1]),
        ]
        transcript = _make_transcript(steps=steps, tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["tool_error_rate"] == 1.0

        passed, score = grader.determine_pass(metrics, task)
        assert passed is False
        assert score == 0.0

    def test_phantom_calls_detected(self, task: Task) -> None:
        grader = TraceConsistencyGrader(
            "consistency",
            expected_tools=["search"],
        )
        tool_calls = [
            ToolCall(tool_name="search", arguments={}, result="ok"),
            ToolCall(tool_name="unexpected_tool", arguments={}, result="ok"),
        ]
        steps = [
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[0]),
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[1]),
            TranscriptStep(step_type=StepType.AGENT_OUTPUT, content="done"),
        ]
        transcript = _make_transcript(steps=steps, tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["phantom_calls"] == 1

        passed, _ = grader.determine_pass(metrics, task)
        assert passed is False

    def test_unused_tool_results_counted(self, task: Task) -> None:
        """Tool calls with results but no subsequent AGENT_OUTPUT step."""
        grader = TraceConsistencyGrader("consistency")
        tool_calls = [
            ToolCall(tool_name="search", arguments={}, result="data"),
            ToolCall(tool_name="fetch", arguments={}, result="more data"),
        ]
        steps = [
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[0]),
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[1]),
            # No AGENT_OUTPUT follows
        ]
        transcript = _make_transcript(steps=steps, tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["unused_tool_results"] == 2

    def test_tool_results_used_when_output_follows(self, task: Task) -> None:
        """Tool call followed by AGENT_OUTPUT is considered used."""
        grader = TraceConsistencyGrader("consistency")
        tool_calls = [
            ToolCall(tool_name="search", arguments={}, result="data"),
        ]
        steps = [
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[0]),
            TranscriptStep(step_type=StepType.AGENT_OUTPUT, content="based on data"),
        ]
        transcript = _make_transcript(steps=steps, tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["unused_tool_results"] == 0

    def test_no_tool_calls_passes(self, task: Task) -> None:
        grader = TraceConsistencyGrader("consistency")
        transcript = _make_transcript()

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["tool_error_rate"] == 0.0
        assert metrics["unused_tool_results"] == 0
        assert metrics["phantom_calls"] == 0

        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == 1.0

    def test_partial_error_rate_passes(self, task: Task) -> None:
        """Error rate below 0.5 should pass."""
        grader = TraceConsistencyGrader("consistency")
        tool_calls = [
            ToolCall(tool_name="search", arguments={}, result="ok"),
            ToolCall(tool_name="search", arguments={}, result="ok"),
            ToolCall(tool_name="search", arguments={}, error="fail"),
        ]
        steps = [
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[0]),
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[1]),
            TranscriptStep(step_type=StepType.TOOL_CALL, tool_call=tool_calls[2]),
            TranscriptStep(step_type=StepType.AGENT_OUTPUT, content="done"),
        ]
        transcript = _make_transcript(steps=steps, tool_calls=tool_calls)

        metrics = grader.compute_metrics(transcript, task)
        assert metrics["tool_error_rate"] == pytest.approx(1 / 3)

        passed, score = grader.determine_pass(metrics, task)
        assert passed is True
        assert score == pytest.approx(2 / 3)

    def test_grade_async(self, task: Task) -> None:
        grader = TraceConsistencyGrader("consistency")
        transcript = _make_transcript()

        outcome = asyncio.run(grader.grade(transcript, task))
        assert outcome.passed is True
        assert outcome.score == 1.0
