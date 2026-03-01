"""Tests for WorkflowRunner and WorkflowAdapter."""

from datetime import datetime
from typing import Any

import pytest

from eval_kit.core.grader import CodeGrader
from eval_kit.core.task import Task, EvalSet
from eval_kit.core.transcript import Transcript, TranscriptStep, StepType
from eval_kit.core.workflow import WorkflowStep, WorkflowTask
from eval_kit.execution.agent_adapter import AgentAdapter
from eval_kit.execution.runner import EvaluationRunner, RunnerConfig
from eval_kit.execution.workflow_runner import WorkflowAdapter, WorkflowRunner


class _DictEchoAdapter(AgentAdapter):
    """Returns task.input_data as the final output."""

    async def run(self, task: Task) -> Transcript:
        transcript = self.start_transcript(task)
        transcript.final_output = dict(task.input_data)
        transcript.add_step(TranscriptStep(
            step_type=StepType.AGENT_OUTPUT,
            content=task.input_data,
        ))
        transcript.completed_at = datetime.utcnow()
        return transcript


class _FailOnStepAdapter(AgentAdapter):
    """Fails on a specific step index (by task_id pattern)."""

    def __init__(self, fail_step: int) -> None:
        self.fail_step = fail_step

    async def run(self, task: Task) -> Transcript:
        if f"step_{self.fail_step}" in task.task_id:
            raise RuntimeError(f"Step {self.fail_step} exploded")
        transcript = self.start_transcript(task)
        transcript.final_output = dict(task.input_data)
        transcript.add_step(TranscriptStep(
            step_type=StepType.AGENT_OUTPUT,
            content=task.input_data,
        ))
        transcript.completed_at = datetime.utcnow()
        return transcript


class _PassGrader(CodeGrader):
    def __init__(self) -> None:
        super().__init__("pass_grader")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"quality": 1.0}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return True, 1.0


class TestWorkflowRunner:
    async def test_two_step_workflow(self):
        """Steps execute in order, second step sees first step's output."""
        adapter = _DictEchoAdapter()
        runner = WorkflowRunner(adapter)

        wt = WorkflowTask(
            task_id="wf1",
            name="Two-Step",
            input_data={},
            steps=[
                WorkflowStep(
                    step_id="s1", name="Step 1",
                    input_data={"goal": "learn python"},
                ),
                WorkflowStep(
                    step_id="s2", name="Step 2",
                    input_data={"ref": "{steps.0.goal}"},
                ),
            ],
        )

        transcript = await runner.run(wt)

        assert transcript.final_output == {"ref": "learn python"}
        assert transcript.metadata["steps_completed"] == 2

    async def test_three_step_chain(self):
        adapter = _DictEchoAdapter()
        runner = WorkflowRunner(adapter)

        wt = WorkflowTask(
            task_id="wf2",
            name="Three-Step",
            input_data={},
            steps=[
                WorkflowStep(step_id="s1", name="S1", input_data={"id": "abc"}),
                WorkflowStep(step_id="s2", name="S2", input_data={"ref_id": "{steps.0.id}", "val": "42"}),
                WorkflowStep(step_id="s3", name="S3", input_data={"combined": "{steps.0.id}-{steps.1.val}"}),
            ],
        )

        transcript = await runner.run(wt)
        assert transcript.final_output == {"combined": "abc-42"}

    async def test_fail_fast_stops_on_failure(self):
        adapter = _FailOnStepAdapter(fail_step=1)
        runner = WorkflowRunner(adapter)

        wt = WorkflowTask(
            task_id="wf3",
            name="Fail Fast",
            input_data={},
            fail_fast=True,
            steps=[
                WorkflowStep(step_id="s1", name="S1", input_data={"x": 1}),
                WorkflowStep(step_id="s2", name="S2", input_data={"y": 2}),
                WorkflowStep(step_id="s3", name="S3", input_data={"z": 3}),
            ],
        )

        transcript = await runner.run(wt)
        # Step 0 completes, step 1 fails, step 2 skipped
        assert transcript.metadata["steps_completed"] == 1
        assert transcript.has_errors

    async def test_no_fail_fast_continues(self):
        adapter = _FailOnStepAdapter(fail_step=1)
        runner = WorkflowRunner(adapter)

        wt = WorkflowTask(
            task_id="wf4",
            name="Continue",
            input_data={},
            fail_fast=False,
            steps=[
                WorkflowStep(step_id="s1", name="S1", input_data={"x": 1}),
                WorkflowStep(step_id="s2", name="S2", input_data={"y": 2}),
                WorkflowStep(step_id="s3", name="S3", input_data={"z": 3}),
            ],
        )

        transcript = await runner.run(wt)
        # Step 0 ok, step 1 fails, step 2 ok (no template refs to step 1)
        assert transcript.metadata["steps_completed"] == 2

    async def test_per_step_grading(self):
        adapter = _DictEchoAdapter()
        grader = _PassGrader()
        runner = WorkflowRunner(adapter, graders={"pass_grader": grader})

        wt = WorkflowTask(
            task_id="wf5",
            name="Graded",
            input_data={},
            steps=[
                WorkflowStep(
                    step_id="s1", name="S1",
                    input_data={"x": 1},
                    grader_ids=["pass_grader"],
                ),
            ],
        )

        transcript = await runner.run(wt)
        assert transcript.metadata["steps_completed"] == 1


class TestWorkflowAdapter:
    async def test_regular_task_passes_through(self):
        inner = _DictEchoAdapter()
        adapter = WorkflowAdapter(inner)

        task = Task(task_id="t1", name="Regular", input_data={"goal": "test"})
        transcript = await adapter.run(task)
        assert transcript.final_output == {"goal": "test"}

    async def test_workflow_task_uses_runner(self):
        inner = _DictEchoAdapter()
        adapter = WorkflowAdapter(inner)

        wt = WorkflowTask(
            task_id="wf1",
            name="Workflow",
            input_data={},
            steps=[
                WorkflowStep(step_id="s1", name="S1", input_data={"a": "1"}),
                WorkflowStep(step_id="s2", name="S2", input_data={"b": "{steps.0.a}"}),
            ],
        )
        transcript = await adapter.run(wt)
        assert transcript.final_output == {"b": "1"}
        assert transcript.metadata["steps_completed"] == 2

    async def test_integration_with_evaluation_runner(self):
        """WorkflowAdapter works with EvaluationRunner for mixed task sets."""
        inner = _DictEchoAdapter()
        adapter = WorkflowAdapter(inner)
        grader = _PassGrader()

        regular = Task(task_id="t1", name="Regular", input_data={"x": 1})
        workflow = WorkflowTask(
            task_id="wf1",
            name="Workflow",
            input_data={},
            steps=[
                WorkflowStep(step_id="s1", name="S1", input_data={"y": 2}),
            ],
        )

        eval_set = EvalSet(name="Mixed", tasks=[regular, workflow])
        runner = EvaluationRunner(adapter, [grader])
        batch = await runner.run(eval_set)

        assert batch.total_count == 2
        assert batch.passed_count == 2
