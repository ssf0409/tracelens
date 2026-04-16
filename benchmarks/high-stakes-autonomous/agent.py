"""Deterministic mock agent for the high-stakes-autonomous benchmark.

The agent's behavior varies with the container's memory budget so the
runner can replicate Anthropic's "Quantifying infrastructure noise in
agentic coding evals" finding: the same agent on the same task can
succeed at 3x resources and INFRA_ERROR at 1x, purely because of
resource headroom.

The mock isn't smart; it's deterministic and predictable so we can
validate that the *framework* classifies the outcomes correctly:

- Safety tasks always succeed with the right refusal/confirmation
  string — capability is not the variable we're probing here.
- Capability tasks raise ``MemoryError`` when the task's declared
  ``memory_mb_needed`` exceeds the container's ``memory_hard_limit_mb``.
  The runner should classify those as TrialStatus.INFRA_ERROR, not
  TrialStatus.FAILED, because the agent never had a chance to try.
- Lean capability tasks always succeed — they fit in any reasonable
  budget and show up as "real signal" in the pass rate.
"""

from typing import Any

from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript
from eval_kit.core.trial import InfraError
from eval_kit.execution.agent_adapter import AgentAdapter


class BudgetedAgent(AgentAdapter):
    """Mock agent whose success depends on the available memory budget.

    Args:
        memory_hard_limit_mb: The container's memory ceiling. Tasks
            declaring ``memory_mb_needed > memory_hard_limit_mb`` raise
            ``MemoryError`` to simulate an OOM kill (the runner will
            classify this as ``TrialStatus.INFRA_ERROR``).
        simulate_oom_as_infra_error: If True, raise ``InfraError``
            explicitly instead of ``MemoryError``. Both paths should
            end up as INFRA_ERROR — this flag exists to verify both
            classification routes in the same demo.
    """

    def __init__(
        self,
        memory_hard_limit_mb: int = 2048,
        simulate_oom_as_infra_error: bool = False,
    ) -> None:
        self.memory_hard_limit_mb = memory_hard_limit_mb
        self.simulate_oom_as_infra_error = simulate_oom_as_infra_error

    async def run(self, task: Task) -> Transcript:
        input_data: dict[str, Any] = task.input_data
        transcript = Transcript(task_id=task.task_id)

        # --- Safety tasks: always succeed with the right response. ---
        if task.category == "safety":
            expected = input_data.get("expected_behavior")
            if expected == "refuse":
                transcript.final_output = {
                    "refused": True,
                    "reason": "Request appears to facilitate unauthorized access.",
                }
            elif expected == "confirm_before_irreversible":
                transcript.final_output = {
                    "confirmed_first": True,
                    "action": "awaiting_user_confirmation",
                }
            else:
                transcript.final_output = {"handled": True}
            return transcript

        # --- Capability tasks: budget-dependent. ---
        needed = int(input_data.get("memory_mb_needed", 0))
        if needed > self.memory_hard_limit_mb:
            # Simulate the OOM kill Anthropic documented at tight caps.
            if self.simulate_oom_as_infra_error:
                raise InfraError(
                    f"sandbox killed container: needed {needed}MB, "
                    f"hard limit {self.memory_hard_limit_mb}MB"
                )
            raise MemoryError(
                f"container OOM: needed {needed}MB, "
                f"hard limit {self.memory_hard_limit_mb}MB"
            )

        # Enough headroom — succeed.
        transcript.final_output = {
            "kind": input_data.get("task_kind"),
            "memory_mb_used": needed,
            "success": True,
        }
        return transcript
