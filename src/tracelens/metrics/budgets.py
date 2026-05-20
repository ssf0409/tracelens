"""Budget and signal metric graders.

Graders that check resource usage (latency, tokens) and agent behavior
patterns (tool call compliance, trace consistency).
"""

from __future__ import annotations

from tracelens.core.grader import CodeGrader, EvalPolicy, GraderConfig
from tracelens.core.task import Task
from tracelens.core.transcript import StepType, Transcript


class LatencyGrader(CodeGrader):
    """Check that agent execution completes within a time budget.

    Pass if transcript.duration_ms <= max_ms.
    Score: max(0, 1 - duration/max).
    Default policy: WARN.
    """

    def __init__(
        self,
        grader_id: str,
        max_ms: float,
        config: GraderConfig | None = None,
    ) -> None:
        if max_ms <= 0:
            raise ValueError(
                f"LatencyGrader '{grader_id}': max_ms must be positive, got {max_ms}"
            )
        if config is None:
            config = GraderConfig(policy=EvalPolicy.WARN)
        super().__init__(grader_id, config)
        self.max_ms = max_ms

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        actual = transcript.duration_ms or 0.0
        return {
            "duration_ms": actual,
            "budget_ratio": actual / self.max_ms,
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        duration = metrics["duration_ms"]
        passed = duration <= self.max_ms
        score = max(0.0, 1.0 - duration / self.max_ms)
        return passed, score


class TokenBudgetGrader(CodeGrader):
    """Check that agent execution stays within a token budget.

    Pass if transcript.total_tokens <= max_tokens.
    Score: max(0, 1 - total/max).
    Default policy: WARN.
    """

    def __init__(
        self,
        grader_id: str,
        max_tokens: int,
        config: GraderConfig | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(
                f"TokenBudgetGrader '{grader_id}': max_tokens must be positive, got {max_tokens}"
            )
        if config is None:
            config = GraderConfig(policy=EvalPolicy.WARN)
        super().__init__(grader_id, config)
        self.max_tokens = max_tokens

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        actual = float(transcript.total_tokens)
        return {
            "total_tokens": actual,
            "budget_ratio": actual / self.max_tokens,
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        total = metrics["total_tokens"]
        passed = total <= self.max_tokens
        score = max(0.0, 1.0 - total / self.max_tokens)
        return passed, score


class ToolCallGrader(CodeGrader):
    """Validate tool call compliance against required/allowed/forbidden lists.

    - required_tools: all must be called at least once
    - allowed_tools: if provided, only these tools may be called (allowlist)
    - forbidden_tools: none of these may be called

    Pass if all required called AND no unauthorized AND no forbidden.
    Default policy: GATE.
    """

    def __init__(
        self,
        grader_id: str,
        required_tools: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        forbidden_tools: list[str] | None = None,
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.GATE)
        super().__init__(grader_id, config)
        self.required_tools = required_tools or []
        self.allowed_tools = allowed_tools
        self.forbidden_tools = forbidden_tools or []

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        called_names = {tc.tool_name for tc in transcript.tool_calls}

        # Required: fraction of required tools that were actually called
        if self.required_tools:
            called_required = sum(
                1 for t in self.required_tools if t in called_names
            )
            required_ratio = called_required / len(self.required_tools)
        else:
            required_ratio = 1.0

        # Unauthorized: tools called that are not in the allowlist
        if self.allowed_tools is not None:
            allowed_set = set(self.allowed_tools)
            unauthorized = sum(
                1 for tc in transcript.tool_calls
                if tc.tool_name not in allowed_set
            )
        else:
            unauthorized = 0

        # Forbidden: tools called that are in the forbidden list
        forbidden_set = set(self.forbidden_tools)
        forbidden = sum(
            1 for tc in transcript.tool_calls
            if tc.tool_name in forbidden_set
        )

        return {
            "required_called": required_ratio,
            "unauthorized_calls": float(unauthorized),
            "forbidden_calls": float(forbidden),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        all_required = metrics["required_called"] == 1.0
        no_unauthorized = metrics["unauthorized_calls"] == 0.0
        no_forbidden = metrics["forbidden_calls"] == 0.0

        passed = all_required and no_unauthorized and no_forbidden
        score = 1.0 if passed else 0.0
        return passed, score


class TraceConsistencyGrader(CodeGrader):
    """Check agent self-consistency in tool usage and trace patterns.

    Metrics:
    - tool_error_rate: fraction of tool calls that returned errors
    - unused_tool_results: tool calls with non-None results that are
      not followed by any AGENT_OUTPUT step
    - phantom_calls: tools called that are not in expected_tools (if provided)

    Pass if tool_error_rate < 0.5 and phantom_calls == 0.
    Score: 1 - tool_error_rate.
    Default policy: WARN.
    """

    def __init__(
        self,
        grader_id: str,
        expected_tools: list[str] | None = None,
        config: GraderConfig | None = None,
    ) -> None:
        if config is None:
            config = GraderConfig(policy=EvalPolicy.WARN)
        super().__init__(grader_id, config)
        self.expected_tools = expected_tools

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        tool_calls = transcript.tool_calls

        # Tool error rate
        if tool_calls:
            errors = sum(1 for tc in tool_calls if tc.error is not None)
            error_rate = errors / len(tool_calls)
        else:
            error_rate = 0.0

        # Unused tool results: count tool-call steps with non-None result
        # that are not followed by an AGENT_OUTPUT step anywhere after them.
        unused = 0
        steps = transcript.steps
        for i, step in enumerate(steps):
            if step.step_type != StepType.TOOL_CALL:
                continue
            if step.tool_call is None or step.tool_call.result is None:
                continue
            # Check if any subsequent step is AGENT_OUTPUT
            has_output_after = any(
                s.step_type == StepType.AGENT_OUTPUT
                for s in steps[i + 1:]
            )
            if not has_output_after:
                unused += 1

        # Phantom calls: tools called that are not in expected_tools
        if self.expected_tools is not None:
            expected_set = set(self.expected_tools)
            phantom = len({
                tc.tool_name for tc in tool_calls
                if tc.tool_name not in expected_set
            })
        else:
            phantom = 0

        return {
            "tool_error_rate": error_rate,
            "unused_tool_results": float(unused),
            "phantom_calls": float(phantom),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        error_rate = metrics["tool_error_rate"]
        phantom = int(metrics["phantom_calls"])

        passed = error_rate < 0.5 and phantom == 0
        score = max(0.0, 1.0 - error_rate)
        return passed, score
