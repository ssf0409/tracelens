"""End-to-end integration test: SimpleAdapter + CodeGrader + EvalSet -> full pipeline."""

import pytest

from tracelens.core.grader import CodeGrader
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import TrialStatus
from tracelens.execution.agent_adapter import SimpleAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig
from tracelens.reporting.generator import ReportGenerator


class _MathGrader(CodeGrader):
    """Grader that checks if the agent's answer matches expected output."""

    def __init__(self):
        super().__init__("math_grader")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = task.input_data.get("expected", 0)
        actual = transcript.final_output.get("answer", 0) if transcript.final_output else 0
        accuracy = 1.0 if actual == expected else 0.0
        return {"accuracy": accuracy}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["accuracy"] >= 1.0, metrics["accuracy"]


async def _math_agent(input_data: dict) -> dict:
    """Trivial agent: adds two numbers."""
    a = input_data.get("a", 0)
    b = input_data.get("b", 0)
    return {"answer": a + b}


class TestEndToEnd:
    async def test_full_pipeline(self):
        """Full pipeline: tasks -> adapter -> runner -> grader -> report."""
        # Define tasks
        tasks = [
            Task(
                task_id="add-1",
                name="1 + 2",
                input_data={"a": 1, "b": 2, "expected": 3},
            ),
            Task(
                task_id="add-2",
                name="10 + 20",
                input_data={"a": 10, "b": 20, "expected": 30},
            ),
            Task(
                task_id="add-wrong",
                name="Wrong expectation",
                input_data={"a": 1, "b": 1, "expected": 999},
            ),
        ]
        eval_set = EvalSet(name="Math Suite", tasks=tasks)

        # Run evaluation
        adapter = SimpleAdapter(_math_agent)
        grader = _MathGrader()
        config = RunnerConfig(num_runs=3)
        runner = EvaluationRunner(adapter, [grader], config)
        batch = await runner.run(eval_set)

        # Verify batch structure
        assert batch.total_count == 9  # 3 tasks × 3 runs
        assert batch.all_complete

        # Verify pass/fail by task
        results = batch.get_pass_results_by_task()
        assert all(results["add-1"])       # All pass
        assert all(results["add-2"])       # All pass
        assert not any(results["add-wrong"])  # All fail

        # Verify trial details
        for trial in batch.trials:
            assert trial.status == TrialStatus.COMPLETED
            assert trial.transcript is not None
            assert len(trial.outcomes) == 1
            # Outcome trial_id matches trial (not task_id)
            assert trial.outcomes[0].trial_id == trial.trial_id

        # Generate report
        gen = ReportGenerator(k_values=[1, 3])
        report = gen.build_report(batch)

        assert report.total_trials == 9
        assert report.total_tasks == 3
        # 2 out of 3 tasks always pass: 6/9 = 66.7%
        assert report.overall_pass_rate == pytest.approx(6 / 9)

        # Per-task summaries
        summaries = {s.task_id: s for s in report.task_summaries}
        assert summaries["add-1"].pass_rate == pytest.approx(1.0)
        assert summaries["add-wrong"].pass_rate == pytest.approx(0.0)

        # Markdown rendering
        md = gen.render_markdown(report)
        assert "Math Suite" not in md  # It's "Evaluation Report" not suite name
        assert "add-1" in md
        assert "add-wrong" in md

        # CI summary
        ci = gen.render_ci_summary(report)
        assert "TraceLens" in ci
        assert "pass_rate=66.7%" in ci
