"""Complete runnable example: load tasks, run evaluation, generate report.

Usage:
    uv run python examples/run_eval.py
"""

import asyncio
import random
import sys
from pathlib import Path
from typing import Any

# Ensure the examples package is importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_kit import (
    CodeGrader,
    EvalSet,
    EvaluationRunner,
    RunnerConfig,
    SimpleAdapter,
    Transcript,
)
from eval_kit.core.task import JSONTaskLoader, Task
from eval_kit.reporting.generator import ReportGenerator

# --- Grader ---

class MathGrader(CodeGrader):
    """Grades math tasks by comparing answer to expected value."""

    def __init__(self) -> None:
        super().__init__(grader_id="math")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = task.metadata["expected"]
        actual = transcript.final_output.get("answer")
        if actual is None:
            return {"correct": 0.0, "error": float("inf")}
        return {"correct": 1.0 if actual == expected else 0.0, "error": abs(actual - expected)}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]


# --- Agent function ---

async def math_agent(input_data: dict[str, Any]) -> dict[str, Any]:
    """A simple math agent that adds or multiplies two numbers.

    Deliberately introduces occasional errors to demonstrate
    non-determinism handling with pass@k and pass^k.
    """
    a = input_data["a"]
    b = input_data["b"]
    operation = input_data.get("operation", "add")

    # Simulate ~10% error rate for non-determinism
    if random.random() < 0.1:
        return {"answer": a + b + 1}  # off-by-one error

    if operation == "multiply":
        return {"answer": a * b}
    return {"answer": a + b}


def main() -> None:
    # 1. Load tasks from JSON
    task_file = Path(__file__).parent / "tasks.json"
    loader = JSONTaskLoader()
    tasks = loader.load(task_file)
    eval_set = EvalSet(name="Math Examples", tasks=tasks)

    print(f"Loaded {len(eval_set.tasks)} tasks from {task_file.name}")

    # 2. Create adapter and grader
    adapter = SimpleAdapter(math_agent)
    grader = MathGrader()

    # 3. Run evaluation (3 runs per task for pass@k / pass^k)
    config = RunnerConfig(num_runs=3, max_concurrency=5, timeout_seconds=30.0)
    runner = EvaluationRunner(adapter, [grader], config)

    print("Running evaluation...")
    batch = asyncio.run(runner.run(eval_set))

    print(f"Completed: {batch.total_count} trials, pass rate: {batch.pass_rate:.1%}")

    # 4. Generate reports
    gen = ReportGenerator(k_values=[1, 3], consistency_k_values=[2, 3])
    report = gen.build_report(batch)

    # Print markdown report
    print("\n" + gen.render_markdown(report))

    # Print CI summary
    print("--- CI Summary ---")
    print(gen.render_ci_summary(report))

    # Save HTML report
    html_path = Path(__file__).parent / "report.html"
    html_content = gen.render_html(report)
    html_path.write_text(html_content)
    print(f"\nHTML report saved to {html_path}")


if __name__ == "__main__":
    main()
