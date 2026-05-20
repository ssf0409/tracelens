# TraceLens / 迹镜

TraceLens is a friendly evaluation and regression-testing framework for AI agents. It turns agent runs into inspectable traces, graded outcomes, baseline comparisons, and CI-ready reliability signals.

迹镜是一个面向 AI Agent 的评测与回归检测框架。它把每次 agent run 转化成可观察的轨迹、可评分的结果、可比较的 baseline，以及可用于 CI 的可靠性信号。

## Overview

TraceLens provides a unified evaluation methodology for AI agent projects. It supports both **subjective evaluations** (LLM-as-judge for quality assessment) and **objective evaluations** (deterministic metrics like schema validity, tool-use constraints, latency, budget, or domain-specific scores).

## Architecture

```
src/tracelens/
├── core/                    # Abstract interfaces
│   ├── task.py              # Task, TaskLoader, EvalSet
│   ├── trial.py             # Trial, TrialBatch execution model
│   ├── grader.py            # Grader ABCs (CodeGrader, LLMGrader, CompositeGrader)
│   ├── transcript.py        # Agent execution logging
│   ├── decision_spec.py     # Reproducibility fingerprinting
│   └── outcome.py           # Grading results
├── execution/               # Trial runner
│   ├── runner.py            # EvaluationRunner - parallel/concurrent execution
│   ├── agent_adapter.py     # AgentAdapter ABC, SimpleAdapter
│   └── registry.py          # Plugin loading via dotted import paths
├── statistics/              # Non-determinism handling
│   ├── pass_at_k.py         # Capability ceiling (pass@k)
│   ├── consistency.py       # Reliability (pass^k)
│   └── inference.py         # Bootstrap CI, significance testing
├── baselines/               # Regression detection
│   ├── manager.py           # Baseline storage, promotion semantics
│   └── comparison.py        # RegressionDetector, severity levels
├── reporting/               # Output
│   └── generator.py         # ReportGenerator (markdown, CI summary, HTML)
└── cli/                     # Command-line interface
    └── main.py              # tracelens run / tracelens report
```

> **Planned modules**: `human_eval/` (sample selection, LLM-human reconciliation) is designed but not yet implemented.

## Core Concepts

### Task

A Task defines a single evaluation test case:

```python
from tracelens import Task

task = Task(
    name="Portfolio website decomposition",
    input_data={
        "goal": "Build a personal portfolio website",
        "user_context": {"experience": "beginner", "hours_per_week": 15}
    },
    category="programming",
    tags=["web", "beginner"],
)
```

### Grader

Graders evaluate agent outputs. There are two main types:

**CodeGrader** - For deterministic metrics:
```python
from tracelens import CodeGrader

class SharpeGrader(CodeGrader):
    def compute_metrics(self, transcript, task):
        returns = transcript.final_output["returns"]
        return {"sharpe_ratio": calculate_sharpe_ratio(returns)}

    def determine_pass(self, metrics, task):
        passed = metrics["sharpe_ratio"] >= 1.0
        score = min(metrics["sharpe_ratio"] / 2.0, 1.0)  # Normalize
        return passed, score
```

**LLMGrader** - For subjective quality (planning, summarisation, helpfulness):
```python
from tracelens import LLMGrader

class SpecificityGrader(LLMGrader):
    def build_grading_prompt(self, transcript, task):
        return f"""Evaluate specificity of this decomposition:
        {transcript.final_output}

        Score 1-10 on: concrete actions, quantifiable targets, named resources
        """

    def parse_llm_response(self, response, task):
        # Parse LLM JSON response
        return passed, score, metrics, feedback
```

### Trial

A Trial represents a single execution of a Task:

```python
from tracelens import Trial, TrialStatus

trial = Trial(
    task_id=task.task_id,
    run_index=0,
    total_runs=5,  # For pass@k
    status=TrialStatus.COMPLETED,
    transcript=transcript,
    outcomes=[outcome1, outcome2],
)
```

### Non-Determinism Handling

**pass@k** - Probability of at least one success in k attempts:
- Use for capability evaluation (can the agent solve this at all?)
- Higher k = higher pass@k (more chances to succeed)

**pass^k** - Probability of all k attempts succeeding:
- Use for reliability evaluation (is the agent consistent?)
- Higher k = lower pass^k (harder to pass every time)

```python
from tracelens.statistics import pass_at_k, pass_to_k

# Capability: can it succeed at least once in 5 tries?
capability = pass_at_k(n=10, c=7, k=5)  # 0.99+

# Reliability: will it succeed every time?
reliability = pass_to_k(results=[True, True, False, True, True], k=3)  # 0.33
```

### Reproducibility with DecisionSpec

`DecisionSpec` captures all parameters affecting agent behavior for reproducibility. The fingerprint is a SHA-256 hash of the entire configuration.

```python
from tracelens.core.decision_spec import DecisionSpec, ModelConfig, AgentSpec

# Capture agent configuration
decision_spec = DecisionSpec(
    model=ModelConfig(
        model_id="gpt-4-turbo",
        temperature=0.7,
        max_tokens=4096,
    ),
    agent=AgentSpec(
        agent_id="goal-decomposer-v2",
        version="1.2.3",
        git_commit="abc123",
    ),
    global_seed=42,
)

# Get fingerprint for reproducibility tracking
print(f"Fingerprint: {decision_spec.fingerprint[:16]}...")

# Attach to transcript for full reproducibility
transcript = Transcript(
    task_id="task-1",
    final_output={"result": "..."},
    decision_spec=decision_spec,
)
```

### Grader Roles (Must-Pass vs Score-Contributor)

Graders can have two roles in composite evaluation:

- **MUST_PASS**: Safety/constraint graders. Any failure = trial fails.
- **SCORE_CONTRIBUTOR**: Quality graders. Contribute to weighted average.

```python
from tracelens import CompositeGrader, GraderRole, GraderConfig

# Safety grader - must pass or entire trial fails
safety_config = GraderConfig(role=GraderRole.MUST_PASS)
safety_grader = FormatValidationGrader("format", config=safety_config)

# Quality grader - contributes to score average
quality_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)
quality_grader = SpecificityGrader("specificity", config=quality_config)

# Composite: safety failure = trial failure, quality affects score
composite = CompositeGrader(
    grader_id="combined",
    graders=[
        (safety_grader, 0.2),   # Weight still affects score
        (quality_grader, 0.8),  # Higher weight for quality
    ],
)

outcome = await composite.grade(transcript, task)
# outcome.passed = False if safety_grader fails, regardless of quality score
```

### Baseline Regression Detection

```python
from tracelens.baselines import BaselineManager, RegressionDetector

manager = BaselineManager("baselines/baselines.json")
baseline = manager.get_baseline("btc_backtest")

detector = RegressionDetector(significance_level=0.05)
report = detector.compare(baseline, current_results)

if report.should_block_ci(threshold=RegressionSeverity.MODERATE):
    sys.exit(1)  # Block the PR
```

### Baseline Promotion (Canary vs Capability)

Baselines can be protected or auto-promoted based on their type:

- **CANARY**: Protected baselines that never auto-update. Manual promotion only.
- **CAPABILITY**: Track improvements over time. Auto-promote when criteria met.
- **EXPERIMENTAL**: For testing. No restrictions.

```python
from tracelens.baselines import BaselineManager, BaselineType, PromotionPolicy

manager = BaselineManager("baselines/baselines.json")

# Create a canary baseline (protected, manual promotion only)
canary = manager.create_canary_baseline(
    task_id="critical_safety_check",
    metrics={"safety_score": 0.95},
)

# Create capability baseline with auto-promotion policy
policy = PromotionPolicy(
    allow_auto_promotion=True,
    min_improvement_relative=0.05,  # 5% improvement required
    min_samples=10,
    required_confidence=0.95,
)
capability = manager.create_capability_baseline(
    task_id="quality_benchmark",
    metrics={"quality_score": 0.75},
    policy=policy,
)

# Try auto-promotion (returns True if promoted)
promoted = manager.try_promote(
    task_id="quality_benchmark",
    new_metrics={"quality_score": 0.82},
    sample_count=15,
)
```

### Statistical Inference (Bootstrap CI)

Research-grade statistical comparison with confidence intervals:

```python
from tracelens.statistics.inference import (
    compare_metrics,
    compare_to_baseline_summary,
    estimate_metric,
)

# Compare current run against baseline with bootstrap CI
baseline_values = [0.72, 0.75, 0.71, 0.74, 0.73]
current_values = [0.78, 0.81, 0.79, 0.82, 0.80]

result = compare_metrics(
    baseline_values,
    current_values,
    confidence=0.95,
    compute_p_value=True,
)

print(f"Baseline: {result.baseline.mean:.3f} ± {result.baseline.std:.3f}")
print(f"Current:  {result.current.mean:.3f} ± {result.current.std:.3f}")
print(f"Difference: {result.difference:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Effect size (Cohen's d): {result.effect_size:.2f}")
print(f"Significant improvement: {result.significant_improvement}")

# Get summary for CI reporting
summary = compare_to_baseline_summary(
    baseline_values,
    current_values,
    metric_name="quality_score",
)
# Returns: "quality_score: 0.800 vs baseline 0.730 (Δ=+0.070, 95% CI [0.045, 0.095], d=1.23, p<0.05)"
```


## CI/CD Integration

### GitHub Actions Workflow

```yaml
- name: Run Evaluation
  run: |
    tracelens run \
      --eval-set eval/suite.json \
      --graders quality,personalization \
      --num-runs 5 \
      --baseline-check \
      --fail-on-regression moderate

- name: Comment on PR
  run: tracelens report --format github-pr
```

### Regression Thresholds

Configure in `baselines/thresholds.py`:

```python
THRESHOLDS = {
    "sharpe_ratio": {
        "direction": "higher_is_better",
        "absolute_threshold": -0.2,  # Block if drops by 0.2
        "relative_threshold": 0.10,   # Block if drops by 10%
    },
    "max_drawdown": {
        "direction": "closer_to_zero_is_better",
        "absolute_threshold": -0.05,
    },
}
```

## Human Evaluation Calibration (Planned)

> The `human_eval/` module is planned but not yet implemented. The recommended workflow:

Weekly process to calibrate LLM graders:

1. **Sample Selection**: Select 20 diverse samples from recent eval runs
2. **Human Rating**: Rate on 1-10 scale per dimension
3. **Correlation Analysis**: Compare LLM vs human scores
4. **Grader Tuning**: Adjust prompts if correlation < 0.7

See [docs/accuracy.md](docs/accuracy.md) for calibration best practices.

## Installation

Install from PyPI:

```bash
# Using uv (recommended)
uv pip install tracelens

# With LLM support
uv pip install "tracelens[llm]"

# Or add to pyproject.toml
# dependencies = [
#     "tracelens>=0.1.0",
# ]
```

### Development Setup

```bash
# Clone and install
git clone https://github.com/ssf0409/tracelens.git
cd tracelens
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run with Docker
docker compose run --rm test
```

## Quick Start

```python
import asyncio
from tracelens import (
    Task, EvalSet, SimpleAdapter, CodeGrader,
    EvaluationRunner, RunnerConfig, Transcript,
)
from tracelens.reporting.generator import ReportGenerator

# 1. Define tasks
tasks = [
    Task(name="Add 2+3", input_data={"a": 2, "b": 3}, metadata={"expected": 5}),
    Task(name="Add 10+20", input_data={"a": 10, "b": 20}, metadata={"expected": 30}),
]
eval_set = EvalSet(name="Math Suite", tasks=tasks)

# 2. Write a simple agent
async def math_agent(input_data: dict) -> dict:
    return {"answer": input_data["a"] + input_data["b"]}

adapter = SimpleAdapter(math_agent)

# 3. Write a grader
class MathGrader(CodeGrader):
    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = task.metadata["expected"]
        actual = transcript.final_output["answer"]
        return {"correct": float(actual == expected)}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]

# 4. Run evaluation
runner = EvaluationRunner(adapter, [MathGrader("math")], RunnerConfig(num_runs=3))
batch = asyncio.run(runner.run(eval_set))

# 5. Generate report
gen = ReportGenerator()
report = gen.build_report(batch)
print(gen.render_markdown(report))
```

> Five-minute version: [`examples/hello_world.py`](examples/hello_world.py).
> Walkthrough: [docs/getting-started.md](docs/getting-started.md).

## Documentation

- **[Getting Started](docs/getting-started.md)** — Run your first eval in five minutes; the example ladder.
- **[Quickstart](docs/quickstart.md)** — Build a custom grader, JSON task loader, and CLI workflow.
- **[Supported Scenarios](docs/scenarios.md)** — Which agent-evaluation problems TraceLens is designed for.
- **[User Guide](docs/user-guide.md)** — Comprehensive framework guide.
- **[Evaluation Levels](docs/evaluation-levels.md)** — Function, task, and system-level evaluation architecture.
- **[Accuracy Best Practices](docs/accuracy.md)** — LLM-judge calibration and grader drift.
- **[CI/CD Integration](docs/ci-cd-integration.md)** — GitHub Actions with regression gating.
- **[Examples](examples/)** — Four working scripts: `hello_world.py` → `contract_eval.py` → `http_agent_eval.py` → `noise_aware_regression.py`.
- **[Releasing](docs/releasing.md)** — Maintainer guide for tag-driven PyPI releases.

## Contributing

TraceLens is MIT licensed and open to contributions. Start with
[CONTRIBUTING.md](CONTRIBUTING.md), run the local verification gate, and open a
focused PR:

```bash
uv run --frozen pytest -q
uv run --frozen ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous
uv run --frozen --extra dev mypy src/tracelens/
```

Security issues should be reported privately using [SECURITY.md](SECURITY.md).

### References

- [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

## Key Design Principles

From Anthropic's evaluation guide:

1. **Grade outcomes, not execution paths** - Focus on what the agent produced
2. **Handle non-determinism with pass@k and pass^k** - Different metrics for capability vs reliability
3. **Start with 20-50 real failure cases** - Build from actual issues
4. **Read transcripts regularly** - Catch false signals and grader bugs
5. **Calibrate with human evaluation** - LLM graders drift without calibration
