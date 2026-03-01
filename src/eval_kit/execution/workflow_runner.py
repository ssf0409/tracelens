"""WorkflowRunner — executes multi-step WorkflowTasks.

Iterates workflow steps in order, resolves inter-step templates,
creates synthetic Tasks per step, and optionally grades each step.
"""

import logging
from datetime import UTC, datetime

from eval_kit.core.grader import Grader
from eval_kit.core.task import Task
from eval_kit.core.transcript import StepType, Transcript, TranscriptStep
from eval_kit.core.workflow import (
    StepResult,
    StepStatus,
    WorkflowContext,
    WorkflowTask,
)
from eval_kit.execution.agent_adapter import AgentAdapter

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Executes WorkflowTasks step-by-step with template resolution.

    Each step creates a synthetic Task, runs the adapter, and optionally
    grades the result. Step outputs feed into subsequent steps via templates.

    Example:
        runner = WorkflowRunner(adapter, graders={"quality": quality_grader})
        transcript = await runner.run(workflow_task)
    """

    def __init__(
        self,
        adapter: AgentAdapter,
        graders: dict[str, Grader] | None = None,
    ) -> None:
        self.adapter = adapter
        self.graders = graders or {}

    async def run(self, workflow_task: WorkflowTask) -> Transcript:
        """Execute all workflow steps and return an aggregated transcript."""
        parent_transcript = Transcript(
            task_id=workflow_task.task_id,
            started_at=datetime.now(UTC),
        )
        context = WorkflowContext()

        for step_idx, step in enumerate(workflow_task.steps):
            step_result = StepResult(
                step_id=step.step_id,
                step_index=step_idx,
            )

            try:
                # Resolve templates from previous step outputs
                resolved_input = context.resolve_input_data(step.input_data)

                # Create synthetic task for this step
                synthetic_task = Task(
                    task_id=f"{workflow_task.task_id}__step_{step_idx}",
                    name=step.name,
                    input_data=resolved_input,
                    expectation=step.expectation,
                    timeout_seconds=step.timeout_seconds,
                )

                # Lifecycle: setup → run → teardown (teardown always called)
                step_transcript: Transcript | None = None
                setup_failed = False

                try:
                    await self.adapter.setup(synthetic_task)
                except Exception as setup_exc:
                    setup_failed = True
                    logger.error(
                        "Setup failed for step '%s': %s",
                        step.step_id,
                        setup_exc,
                    )

                if not setup_failed:
                    try:
                        step_transcript = await self.adapter.run(synthetic_task)
                    finally:
                        try:
                            await self.adapter.teardown(
                                synthetic_task, step_transcript
                            )
                        except Exception as teardown_exc:
                            logger.error(
                                "Teardown failed for step '%s': %s",
                                step.step_id,
                                teardown_exc,
                            )
                            if step_transcript is not None:
                                raise RuntimeError(
                                    f"Teardown failed for step "
                                    f"'{step.step_id}': {teardown_exc}"
                                ) from teardown_exc
                else:
                    # Teardown still called even after setup failure
                    try:
                        await self.adapter.teardown(
                            synthetic_task, step_transcript
                        )
                    except Exception as teardown_exc:
                        logger.error(
                            "Teardown failed for step '%s' "
                            "(after setup failure): %s",
                            step.step_id,
                            teardown_exc,
                        )
                    raise RuntimeError(
                        f"Setup failed for step '{step.step_id}'"
                    )

            except Exception as exc:
                step_result.status = StepStatus.FAILED
                step_result.error = str(exc)

                parent_transcript.add_step(TranscriptStep(
                    step_type=StepType.ERROR,
                    error=f"Workflow step '{step.step_id}' failed: {exc}",
                    content={
                        "workflow_step": step.step_id,
                        "step_index": step_idx,
                        "status": "failed",
                    },
                ))

                context.step_results.append(step_result)

                if workflow_task.fail_fast:
                    break
                continue

            step_result.status = StepStatus.COMPLETED
            step_result.output = step_transcript.final_output
            step_result.transcript = step_transcript

            # Record step in parent transcript
            parent_transcript.add_step(TranscriptStep(
                step_type=StepType.INTERNAL,
                content={
                    "workflow_step": step.step_id,
                    "step_index": step_idx,
                    "status": "completed",
                },
            ))

            # Merge step's transcript steps into parent
            for child_step in step_transcript.steps:
                parent_transcript.add_step(child_step)

            # Per-step grading (outside adapter try/except — grading failure
            # should not mark the step as FAILED since the adapter succeeded)
            if step.grader_ids:
                for gid in step.grader_ids:
                    if gid in self.graders:
                        try:
                            await self.graders[gid].grade(
                                step_transcript, synthetic_task
                            )
                        except Exception as grading_exc:
                            logger.error(
                                "Grader '%s' failed on step '%s': %s",
                                gid,
                                step.step_id,
                                grading_exc,
                            )
                    else:
                        logger.warning(
                            "Grader '%s' requested by step '%s' not found, skipping",
                            gid,
                            step.step_id,
                        )

            context.step_results.append(step_result)

        # Set final output from last completed step
        completed = [r for r in context.step_results if r.status == StepStatus.COMPLETED]
        if completed:
            parent_transcript.final_output = completed[-1].output

        parent_transcript.completed_at = datetime.now(UTC)
        parent_transcript.metadata["workflow_steps"] = len(workflow_task.steps)
        parent_transcript.metadata["steps_completed"] = len(completed)
        parent_transcript.metadata["step_results"] = [
            {"step_id": r.step_id, "status": r.status.value}
            for r in context.step_results
        ]

        return parent_transcript


class WorkflowAdapter(AgentAdapter):
    """Adapter that delegates WorkflowTasks to WorkflowRunner.

    For regular Tasks, passes through to the inner adapter.
    For WorkflowTasks, uses WorkflowRunner for multi-step execution.

    This lets WorkflowTasks flow through the existing EvaluationRunner
    without any runner modifications.
    """

    def __init__(
        self,
        inner: AgentAdapter,
        graders: dict[str, Grader] | None = None,
    ) -> None:
        self.inner = inner
        self.workflow_runner = WorkflowRunner(inner, graders)

    async def run(self, task: Task) -> Transcript:
        if isinstance(task, WorkflowTask):
            return await self.workflow_runner.run(task)
        return await self.inner.run(task)

    async def setup(self, task: Task) -> None:
        # Setup is handled per-step inside WorkflowRunner for WorkflowTasks
        if not isinstance(task, WorkflowTask):
            await self.inner.setup(task)

    async def teardown(self, task: Task, transcript: Transcript | None) -> None:
        if not isinstance(task, WorkflowTask):
            await self.inner.teardown(task, transcript)
