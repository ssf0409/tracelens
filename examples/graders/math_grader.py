"""Example CodeGrader for math evaluation tasks.

A deterministic grader that compares the agent's numerical answer
against an expected value stored in task metadata.
"""

from eval_kit import CodeGrader
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript


class MathGrader(CodeGrader):
    """Grades math tasks by comparing computed answer to expected value.

    Expects:
        - task.metadata["expected"]: the correct answer (number)
        - transcript.final_output["answer"]: the agent's answer (number)

    Produces metrics:
        - correct: 1.0 if exact match, 0.0 otherwise
        - error: absolute difference between answer and expected
    """

    def __init__(self) -> None:
        super().__init__(grader_id="math")

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        expected = task.metadata["expected"]
        actual = transcript.final_output.get("answer")

        if actual is None:
            return {"correct": 0.0, "error": float("inf")}

        return {
            "correct": 1.0 if actual == expected else 0.0,
            "error": abs(actual - expected),
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        passed = metrics["correct"] == 1.0
        score = metrics["correct"]
        return passed, score
