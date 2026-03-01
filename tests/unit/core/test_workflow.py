"""Tests for WorkflowTask and WorkflowContext template resolution."""

import pytest

from eval_kit.core.workflow import (
    StepResult,
    StepStatus,
    WorkflowContext,
    WorkflowStep,
    WorkflowTask,
)


class TestWorkflowTask:
    def test_workflow_task_is_a_task(self):
        wt = WorkflowTask(
            task_id="wf1",
            name="Multi-step",
            input_data={"goal": "test"},
            steps=[
                WorkflowStep(step_id="s1", name="Step 1", input_data={"x": 1}),
                WorkflowStep(step_id="s2", name="Step 2", input_data={"y": 2}),
            ],
        )
        assert wt.task_id == "wf1"
        assert len(wt.steps) == 2
        assert wt.fail_fast is True

    def test_workflow_step_defaults(self):
        step = WorkflowStep(step_id="s1", name="Step", input_data={})
        assert step.timeout_seconds == 300.0
        assert step.grader_ids is None
        assert step.expectation is None


class TestWorkflowContext:
    def _ctx_with_results(self, *outputs: dict) -> WorkflowContext:
        ctx = WorkflowContext()
        for i, output in enumerate(outputs):
            ctx.step_results.append(StepResult(
                step_id=f"s{i}",
                step_index=i,
                status=StepStatus.COMPLETED,
                output=output,
            ))
        return ctx

    def test_simple_template(self):
        ctx = self._ctx_with_results({"goal_id": "g123"})
        result = ctx.resolve_template("Use goal {steps.0.goal_id}")
        assert result == "Use goal g123"

    def test_nested_dict_path(self):
        ctx = self._ctx_with_results({"data": {"items": {"name": "test"}}})
        result = ctx.resolve_template("{steps.0.data.items.name}")
        assert result == "test"

    def test_list_index(self):
        ctx = self._ctx_with_results({"items": ["a", "b", "c"]})
        result = ctx.resolve_template("{steps.0.items.1}")
        assert result == "b"

    def test_multiple_templates(self):
        ctx = self._ctx_with_results({"x": "hello"}, {"y": "world"})
        result = ctx.resolve_template("{steps.0.x} {steps.1.y}")
        assert result == "hello world"

    def test_no_templates(self):
        ctx = WorkflowContext()
        assert ctx.resolve_template("plain text") == "plain text"

    def test_missing_step_raises(self):
        ctx = WorkflowContext()
        with pytest.raises(ValueError, match="step 5"):
            ctx.resolve_template("{steps.5.field}")

    def test_failed_step_raises(self):
        ctx = WorkflowContext()
        ctx.step_results.append(StepResult(
            step_id="s0",
            step_index=0,
            status=StepStatus.FAILED,
            error="boom",
        ))
        with pytest.raises(ValueError, match="status=failed"):
            ctx.resolve_template("{steps.0.field}")

    def test_missing_field_raises(self):
        ctx = self._ctx_with_results({"a": 1})
        with pytest.raises(ValueError, match="Cannot resolve"):
            ctx.resolve_template("{steps.0.missing_key}")

    def test_resolve_input_data_deep(self):
        ctx = self._ctx_with_results({"id": "abc"}, {"name": "test"})
        resolved = ctx.resolve_input_data({
            "target_id": "{steps.0.id}",
            "label": "{steps.1.name}",
            "nested": {"ref": "{steps.0.id}"},
            "list_val": ["{steps.1.name}", "static"],
            "number": 42,
        })
        assert resolved == {
            "target_id": "abc",
            "label": "test",
            "nested": {"ref": "abc"},
            "list_val": ["test", "static"],
            "number": 42,
        }
