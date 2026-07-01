# TraceLens / 迹镜

TraceLens is a friendly evaluation and regression-testing framework for AI agents. It turns agent runs into inspectable traces, graded outcomes, baseline comparisons, and CI-ready reliability signals.

迹镜是一个面向 AI Agent 的评测与回归检测框架。它把每次 agent run 转化成可观察的轨迹、可评分的结果、可比较的 baseline，以及可用于 CI 的可靠性信号。

📖 **Documentation:** <https://ssf0409.github.io/tracelens/> &nbsp;•&nbsp; 📦 **PyPI:** `pip install tracelens`

## Why TraceLens

Agents are non-deterministic. Unit tests are not enough. TraceLens helps teams capture agent traces, grade outcomes, compare against baselines, and block regressions in CI.

Use it when you need to answer questions like:

- Did this agent produce the right outcome, not just run without crashing?
- Is a flaky success still a real capability after 3–5 attempts?
- Did a prompt, model, tool, or infra change regress a baseline?
- Can CI block unsafe or lower-quality agent behavior before it ships?

It supports both **subjective** evaluation (LLM-as-judge for quality) and **objective** evaluation (schema validity, tool-use constraints, latency, budget, or domain-specific metrics) — and keeps harness failures separate from agent failures so a broken eval never looks like a regression.

## Install

```bash
# Recommended: uv
uv pip install tracelens

# Or: plain pip
pip install tracelens
```

For the repository examples and local development tools:

```bash
git clone https://github.com/ssf0409/tracelens.git
cd tracelens
uv pip install -e ".[dev]"
```

See [Installation](https://ssf0409.github.io/tracelens/installation/) for extras (`[llm]`, `[http]`) and CI setup.

## 5-Minute Demo

```bash
python examples/hello_world.py
tracelens report --results examples/reports/hello_world_report.json --format markdown
```

Expected first output:

```text
tracelens hello-world
--------------------
trials run : 9
pass rate  : 100%
report json: examples/reports/hello_world_report.json
sample md  : examples/reports/hello_world_report.md
```

The checked-in [sample report](examples/reports/hello_world_report.md) shows the concrete pieces a real eval needs: tasks, trials, pass@k, pass^k, graders, baseline comparison, regression result, and CI summary.

To start inside your own project:

```bash
tracelens init .
tracelens run \
  --eval-set eval/tasks.json \
  --adapter eval.adapter.StarterAdapter \
  --graders eval.grader.StarterGrader
```

`tracelens init` writes user-owned starter files under `eval/` plus a GitHub Actions workflow. It refuses to overwrite generated files unless you pass `--force`.

## What an eval looks like

Four pieces — Task, Adapter, Grader, Runner — and a report:

```python
import asyncio
from tracelens import (
    Task, EvalSet, SimpleAdapter, CodeGrader,
    EvaluationRunner, RunnerConfig, Transcript,
)
from tracelens.reporting.generator import ReportGenerator

# 1. Define tasks
eval_set = EvalSet(name="Math Suite", tasks=[
    Task(name="Add 2+3", input_data={"a": 2, "b": 3}, metadata={"expected": 5}),
    Task(name="Add 10+20", input_data={"a": 10, "b": 20}, metadata={"expected": 30}),
])

# 2. Wrap your agent
async def math_agent(input_data: dict) -> dict:
    return {"answer": input_data["a"] + input_data["b"]}

adapter = SimpleAdapter(math_agent)

# 3. Write a grader
class MathGrader(CodeGrader):
    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"correct": float(transcript.final_output["answer"] == task.metadata["expected"])}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]

# 4. Run and report
batch = asyncio.run(EvaluationRunner(adapter, [MathGrader("math")], RunnerConfig(num_runs=3)).run(eval_set))
print(ReportGenerator().render_markdown(ReportGenerator().build_report(batch)))
```

> Walkthrough: [Getting Started (5 min)](https://ssf0409.github.io/tracelens/getting-started/).
> Ready for a non-toy agent? [Evaluating a Real Agent](https://ssf0409.github.io/tracelens/real-agent/).

## Documentation

The full, searchable docs live at **<https://ssf0409.github.io/tracelens/>**. Highlights:

| Start here | Concepts | Guides |
|------------|----------|--------|
| [Is TraceLens For You?](docs/scenarios.md) | [Core Concepts & Glossary](docs/concepts.md) | [Evaluating a Real Agent](docs/real-agent.md) |
| [Getting Started (5 min)](docs/getting-started.md) | [pass@k vs pass^k](docs/pass-at-k-vs-pass-hat-k.md) | [Baseline Regression Tutorial](docs/baseline-regression-tutorial.md) |
| [TraceLens vs Adjacent Tools](docs/comparison.md) | [Accuracy Best Practices](docs/accuracy.md) | [Human-Eval Calibration](docs/human-eval.md) |
| [Installation](docs/installation.md) | [Multi-Level Evaluation](docs/evaluation-levels.md) | [CI/CD Integration](docs/ci-cd-integration.md) |

Also: [Build Your First Eval](docs/quickstart.md) · [User Guide](docs/user-guide.md) · [Evaluation Recipes](docs/evaluation-recipes.md) · [API Reference](https://ssf0409.github.io/tracelens/reference/) · [Examples](examples/) · [Roadmap](ROADMAP.md) · [Contributor Testing](docs/contributor-testing.md) · [Releasing](docs/releasing.md).

## Contributing

TraceLens is MIT licensed and open to contributions. Start with [CONTRIBUTING.md](CONTRIBUTING.md), then run the local verification gate:

```bash
make verify   # lock check -> lint -> typecheck -> tests + coverage
```

Security issues should be reported privately using [SECURITY.md](SECURITY.md).

## Key Design Principles

1. **Grade outcomes, not execution paths** — focus on what the agent produced.
2. **Handle non-determinism** — pass@k for capability, pass^k for reliability.
3. **Start with 20–50 real failure cases** — build suites from actual issues.
4. **Read transcripts regularly** — catch false signals and grader bugs.
5. **Calibrate with human evaluation** — LLM graders drift without it.
6. **Separate harness failures from agent failures** — track infra/grader error rates alongside pass rates.

Informed by Anthropic's [Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
