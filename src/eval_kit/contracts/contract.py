"""BehaviorContract: declarative agent contract that auto-generates graders.

A BehaviorContract describes *what* an agent must do (output shape, tool rules,
budget limits, content constraints) without specifying *how* to check it.
The ``to_graders()`` method translates the contract into a list of deterministic
``(Grader, EvalPolicy)`` pairs that can be fed to a ``CompositeGrader`` or
``EvaluationRunner``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from eval_kit.core.grader import EvalPolicy, Grader
from eval_kit.metrics.budgets import LatencyGrader, TokenBudgetGrader, ToolCallGrader
from eval_kit.metrics.validators import ConstraintGrader, ContainsGrader, JsonSchemaGrader


class BehaviorContract(BaseModel):
    """Verifiable contract for agent behavior."""

    contract_id: str
    version: str

    # Output contract
    output_schema: dict[str, Any] | None = None
    output_model: str | None = None  # Pydantic model dotted path

    # Tool contract
    tools_allowed: list[str] = []
    tools_required: list[str] = []
    tool_param_constraints: dict[str, Any] = {}

    # Budget contract
    max_tokens: int | None = None
    max_latency_ms: float | None = None
    max_cost_usd: float | None = None

    # Safety / content contract
    must_include: list[str] = []
    must_not_include: list[str] = []
    custom_constraints: list[dict[str, Any]] = []

    def to_graders(self) -> list[tuple[Grader, EvalPolicy]]:
        """Auto-generate a grader suite from this contract.

        Each non-empty contract section produces one grader with an
        appropriate default policy:
        - output_schema  -> JsonSchemaGrader  (GATE)
        - tools_*        -> ToolCallGrader    (GATE)
        - max_latency_ms -> LatencyGrader     (WARN)
        - max_tokens     -> TokenBudgetGrader (WARN)
        - must_include/must_not_include -> ContainsGrader (TRACK)
        - custom_constraints -> ConstraintGrader (GATE)
        """
        result: list[tuple[Grader, EvalPolicy]] = []
        prefix = self.contract_id

        if self.output_schema is not None:
            grader = JsonSchemaGrader(
                f"{prefix}.json_schema",
                schema=self.output_schema,
            )
            result.append((grader, grader.policy))

        if self.tools_allowed or self.tools_required:
            grader = ToolCallGrader(
                f"{prefix}.tool_call",
                required_tools=self.tools_required or None,
                allowed_tools=self.tools_allowed or None,
            )
            result.append((grader, grader.policy))

        if self.max_latency_ms is not None:
            grader = LatencyGrader(
                f"{prefix}.latency",
                max_ms=self.max_latency_ms,
            )
            result.append((grader, grader.policy))

        if self.max_tokens is not None:
            grader = TokenBudgetGrader(
                f"{prefix}.token_budget",
                max_tokens=self.max_tokens,
            )
            result.append((grader, grader.policy))

        if self.must_include or self.must_not_include:
            grader = ContainsGrader(
                f"{prefix}.contains",
                required=self.must_include,
                forbidden=self.must_not_include or None,
            )
            result.append((grader, grader.policy))

        if self.custom_constraints:
            grader = ConstraintGrader(
                f"{prefix}.constraint",
                constraints=self.custom_constraints,
            )
            result.append((grader, grader.policy))

        return result
