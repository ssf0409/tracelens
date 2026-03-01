"""WorkflowTask — multi-step task type with inter-step data flow.

A WorkflowTask defines a sequence of steps where each step can reference
outputs from previous steps via template strings like {steps.0.goal_id}.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from eval_kit.core.task import Task, TaskExpectation
from eval_kit.core.transcript import Transcript


class StepStatus(str, Enum):
    """Status of a workflow step execution."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep(BaseModel):
    """A single step in a multi-step workflow."""

    step_id: str
    name: str
    input_data: dict[str, Any]
    expectation: TaskExpectation | None = None
    grader_ids: list[str] | None = None
    timeout_seconds: float = 300.0


class StepResult(BaseModel):
    """Result of executing one workflow step."""

    step_id: str
    step_index: int
    status: StepStatus = StepStatus.PENDING
    output: Any = None
    transcript: Transcript | None = None
    error: str | None = None


class WorkflowTask(Task):
    """A task consisting of multiple ordered steps with data flow between them.

    Steps can reference previous step outputs via template strings:
        {steps.0.goal_id} — the `goal_id` field from step 0's output
        {steps.1.data.items.0} — nested access into step 1's output
    """

    steps: list[WorkflowStep] = Field(default_factory=list)
    fail_fast: bool = True


# Template pattern: {steps.N.dotted.path}
_TEMPLATE_RE = re.compile(r"\{steps\.(\d+)\.([^}]+)\}")


class WorkflowContext(BaseModel):
    """Tracks step results and resolves inter-step template references."""

    step_results: list[StepResult] = Field(default_factory=list)

    def resolve_template(self, value: str) -> str:
        """Resolve template strings like {steps.0.field_path} using completed step outputs.

        Navigates dotted paths into step output (dict keys, list indices, object attributes).
        Raises ValueError on unresolvable references — fail-fast for clear debugging.

        TODO: This is a meaningful design choice. Implement to match your preferences.
        Options:
          - Raise immediately (strict, fail-fast) ← current behavior
          - Return placeholder like '<UNRESOLVED: steps.3.goal_id>' (lenient)
          - Make configurable per WorkflowTask
        """
        def _replacer(match: re.Match[str]) -> str:
            step_idx = int(match.group(1))
            field_path = match.group(2)

            if step_idx >= len(self.step_results):
                raise ValueError(
                    f"Template references step {step_idx}, "
                    f"but only {len(self.step_results)} steps have completed"
                )

            result = self.step_results[step_idx]
            if result.status != StepStatus.COMPLETED or result.output is None:
                raise ValueError(
                    f"Template references step {step_idx}, "
                    f"but it has status={result.status.value}"
                )

            current = result.output
            for part in field_path.split("."):
                try:
                    if isinstance(current, dict):
                        current = current[part]
                    elif isinstance(current, (list, tuple)):
                        current = current[int(part)]
                    else:
                        current = getattr(current, part)
                except (KeyError, IndexError, AttributeError, ValueError) as exc:
                    raise ValueError(
                        f"Cannot resolve '{field_path}' in step {step_idx} output: {exc}"
                    ) from exc

            return str(current)

        return _TEMPLATE_RE.sub(_replacer, value)

    def resolve_input_data(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Deep-resolve all template strings in an input_data dict."""
        return self._resolve_recursive(input_data)  # type: ignore[return-value]

    def _resolve_recursive(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.resolve_template(obj)
        if isinstance(obj, dict):
            return {k: self._resolve_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_recursive(item) for item in obj]
        return obj
