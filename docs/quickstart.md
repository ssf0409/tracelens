# Quickstart: eval-kit in 10 Minutes

Get a working evaluation pipeline running from scratch.

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## 1. Install eval-kit

```bash
# From GitHub
uv pip install git+https://github.com/ssf0409/eval-kit.git

# Or for development
git clone https://github.com/ssf0409/eval-kit.git
cd eval-kit
uv pip install -e ".[dev]"
```

## 2. Define Tasks

Create a file `tasks.json` with your evaluation tasks:

```json
{
  "tasks": [
    {
      "task_id": "math-add",
      "name": "Simple addition",
      "input_data": {"a": 2, "b": 3},
      "category": "arithmetic",
      "tags": ["addition", "easy"],
      "metadata": {"expected": 5}
    },
    {
      "task_id": "math-multiply",
      "name": "Multiplication",
      "input_data": {"a": 7, "b": 8},
      "category": "arithmetic",
      "tags": ["multiplication"],
      "metadata": {"expected": 56, "operation": "multiply"}
    }
  ]
}
```

Each task needs:
- `name` and `input_data` (required) — what to test
- `metadata` — any data graders need (e.g., expected answers)
- `tags`, `category`, `difficulty` — for filtering and organization

> See `examples/tasks.json` for a complete 5-task example.

## 3. Write a Simple Agent

An agent is any async function that takes input and returns output:

```python
from typing import Any

async def math_agent(input_data: dict[str, Any]) -> dict[str, Any]:
    a = input_data["a"]
    b = input_data["b"]
    operation = input_data.get("operation", "add")

    if operation == "multiply":
        return {"answer": a * b}
    return {"answer": a + b}
```

Wrap it as an adapter:

```python
from eval_kit import SimpleAdapter

adapter = SimpleAdapter(math_agent)
```

For complex agents, subclass `AgentAdapter` directly — see the [User Guide](user-guide.md#agent-adapters).

## 4. Write a Grader

A `CodeGrader` computes metrics and determines pass/fail:

```python
from eval_kit import CodeGrader
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript


class MathGrader(CodeGrader):
    def __init__(self) -> None:
        super().__init__(grader_id="math")

    def compute_metrics(
        self, transcript: Transcript, task: Task
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
        self, metrics: dict[str, float], task: Task
    ) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]
```

> See `examples/graders/` for both `CodeGrader` and `LLMGrader` examples.

## 5. Run Evaluation

```python
import asyncio
from eval_kit import EvalSet, EvaluationRunner, RunnerConfig
from eval_kit.core.task import JSONTaskLoader

# Load tasks
loader = JSONTaskLoader()
tasks = loader.load("tasks.json")
eval_set = EvalSet(name="Math Suite", tasks=tasks)

# Run with 3 attempts per task (for pass@k statistics)
config = RunnerConfig(num_runs=3, max_concurrency=5, timeout_seconds=30.0)
runner = EvaluationRunner(adapter, [MathGrader()], config)
batch = asyncio.run(runner.run(eval_set))

print(f"Pass rate: {batch.pass_rate:.1%}")
```

## 6. Generate Report

```python
from eval_kit.reporting.generator import ReportGenerator

gen = ReportGenerator(k_values=[1, 3], consistency_k_values=[2, 3])
report = gen.build_report(batch)

# Markdown report
print(gen.render_markdown(report))

# HTML dashboard
with open("report.html", "w") as f:
    f.write(gen.render_html(report))
```

## 7. Interpret Results

The report includes two key statistics:

| Metric | Measures | Higher k means... |
|--------|----------|-------------------|
| **pass@k** | Capability — "can the agent do this at all?" | Higher values (more chances) |
| **pass^k** | Reliability — "does the agent do this consistently?" | Lower values (harder to be perfect) |

- **pass@3 = 0.95** means: "95% chance at least 1 of 3 attempts succeeds" — strong capability
- **pass^3 = 0.60** means: "60% chance all 3 attempts succeed" — moderate reliability

If pass@k is high but pass^k is low, the agent is capable but inconsistent. Increase `num_runs` and investigate failure patterns.

## 8. Use the CLI

Run evaluations from the command line:

```bash
# Run and print CI summary
eval-kit run \
  --eval-set tasks.json \
  --adapter myproject.adapters.MathAdapter \
  --graders myproject.graders.MathGrader \
  --num-runs 3 \
  --report report.md \
  --html-report report.html

# Generate report from saved results
eval-kit report --results results.json --format html
```

## Next Steps

- **[User Guide](user-guide.md)** — Deep dive into all framework components
- **[Accuracy Best Practices](accuracy.md)** — Improve evaluation reliability
- **[Examples](../examples/)** — Working scripts you can run immediately
