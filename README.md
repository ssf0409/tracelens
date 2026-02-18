# Agent Evaluation Framework

A common evaluation framework for AI agents with support for LLM-based and code-based grading, statistical analysis, baseline regression detection, and CI/CD integration.

## Overview

This framework provides a unified evaluation methodology for AI agent projects. It is designed to support both **subjective evaluations** (LLM-as-judge for quality assessment) and **objective evaluations** (deterministic metrics like Sharpe ratio).

### Supported Projects

- **StrideAI** - Goal decomposition agent evaluation (LLM-based quality graders)
- **crypto-trading-system** - Algorithmic trading agent evaluation (code-based financial metrics)

## Architecture

```
agent-eval/
├── core/                    # Abstract interfaces
│   ├── task.py              # Task, TaskLoader, EvalSet ABCs
│   ├── trial.py             # Trial execution model
│   ├── grader.py            # Grader ABC hierarchy (CodeGrader, LLMGrader, CompositeGrader)
│   ├── transcript.py        # Agent execution logging
│   ├── decision_spec.py     # Reproducibility fingerprinting
│   └── outcome.py           # Grading results
├── execution/               # Trial runner
│   ├── runner.py            # Parallel/concurrent execution
│   └── agent_adapter.py     # Agent invocation interface
├── statistics/              # Non-determinism handling
│   ├── pass_at_k.py         # Capability ceiling (pass@k)
│   ├── consistency.py       # Reliability (pass^k)
│   └── inference.py         # Bootstrap CI, significance testing
├── baselines/               # Regression detection
│   ├── manager.py           # Baseline storage, promotion semantics
│   └── comparison.py        # Regression detection
├── human_eval/              # Weekly calibration
│   ├── sampler.py           # Sample selection
│   └── reconciliation.py    # LLM-human comparison
└── reporting/               # Output
    └── ci_output.py         # CI-friendly reporting
```

## Core Concepts

### Task

A Task defines a single evaluation test case:

```python
from agent_eval import Task

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

**CodeGrader** - For deterministic metrics (crypto-trading):
```python
from agent_eval import CodeGrader

class SharpeGrader(CodeGrader):
    def compute_metrics(self, transcript, task):
        returns = transcript.final_output["returns"]
        return {"sharpe_ratio": calculate_sharpe_ratio(returns)}

    def determine_pass(self, metrics, task):
        passed = metrics["sharpe_ratio"] >= 1.0
        score = min(metrics["sharpe_ratio"] / 2.0, 1.0)  # Normalize
        return passed, score
```

**LLMGrader** - For subjective quality (StrideAI):
```python
from agent_eval import LLMGrader

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
from agent_eval import Trial, TrialStatus

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
from agent_eval.statistics import pass_at_k, pass_to_k

# Capability: can it succeed at least once in 5 tries?
capability = pass_at_k(n=10, c=7, k=5)  # 0.99+

# Reliability: will it succeed every time?
reliability = pass_to_k(results=[True, True, False, True, True], k=3)  # 0.33
```

### Reproducibility with DecisionSpec

`DecisionSpec` captures all parameters affecting agent behavior for reproducibility. The fingerprint is a SHA-256 hash of the entire configuration.

```python
from agent_eval.core.decision_spec import DecisionSpec, ModelConfig, AgentSpec

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
from agent_eval import CompositeGrader, GraderRole, GraderConfig

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
from agent_eval.baselines import BaselineManager, RegressionDetector

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
from agent_eval.baselines import BaselineManager, BaselineType, PromotionPolicy

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
from agent_eval.statistics.inference import (
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

## Integration Patterns

### StrideAI Integration

```python
# eval/graders/quality_grader.py
from agent_eval import LLMGrader

class DecompositionQualityGrader(LLMGrader):
    """Evaluates goal decomposition quality using LLM-as-judge."""

    DIMENSIONS = [
        ("specificity", "Are tasks concrete and actionable?"),
        ("personalization", "Is the plan tailored to user context?"),
        ("actionability", "Can the user start immediately?"),
    ]

    def build_grading_prompt(self, transcript, task):
        # Build rubric-based prompt
        ...
```

### Crypto-Trading Integration

```python
# evaluation/graders/financial.py
from agent_eval import CodeGrader

class FinancialGrader(CodeGrader):
    """Wraps existing financial metrics as a grader."""

    def compute_metrics(self, transcript, task):
        from crypto_trading_system.evaluation.metrics import (
            calculate_sharpe_ratio,
            calculate_sortino_ratio,
            calculate_max_drawdown,
        )

        returns = transcript.final_output["returns"]
        return {
            "sharpe_ratio": calculate_sharpe_ratio(returns),
            "sortino_ratio": calculate_sortino_ratio(returns),
            "max_drawdown": calculate_max_drawdown(returns),
        }
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
- name: Run Evaluation
  run: |
    agent-eval run \
      --eval-set eval/suite.json \
      --graders quality,personalization \
      --num-runs 5 \
      --baseline-check \
      --fail-on-regression moderate

- name: Comment on PR
  run: agent-eval report --format github-pr
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

## Human Evaluation Calibration

Weekly process to calibrate LLM graders:

1. **Sample Selection**: Auto-select 20 diverse samples
2. **Human Rating**: Rate on 1-10 scale per dimension
3. **Correlation Analysis**: Compare LLM vs human scores
4. **Grader Tuning**: Adjust prompts if correlation < 0.7

```python
from agent_eval.human_eval import HumanEvalSampler, Reconciler

sampler = HumanEvalSampler(strategy="diverse", sample_size=20)
samples = sampler.select(trials)

# After human evaluation...
reconciler = Reconciler()
report = reconciler.analyze(human_grades, llm_grades)
print(f"Correlation: {report.score_correlation:.2f}")
```

## Installation

Since this is a private package, install directly from GitHub:

```bash
# Using uv (recommended)
uv pip install git+https://github.com/ssf0409/agent-eval.git

# With LLM support
uv pip install "agent-eval[llm] @ git+https://github.com/ssf0409/agent-eval.git"

# Or add to pyproject.toml
# dependencies = [
#     "agent-eval @ git+https://github.com/ssf0409/agent-eval.git",
# ]
```

### Development Setup

```bash
# Clone and install
git clone https://github.com/ssf0409/agent-eval.git
cd agent-eval
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v

# Run with Docker
docker compose run --rm test
```

## Quick Start

```python
from agent_eval import Task, EvalSet, Trial
from agent_eval.execution import EvaluationRunner

# Define tasks
tasks = [
    Task(name="Task 1", input_data={"goal": "..."}),
    Task(name="Task 2", input_data={"goal": "..."}),
]

# Create eval set
eval_set = EvalSet(name="My Suite", tasks=tasks)

# Run evaluation
runner = EvaluationRunner(
    graders=[my_grader],
    adapter=my_agent_adapter,
    num_runs=5,
)
results = await runner.run(eval_set)

# Analyze
print(f"Pass rate: {results.pass_rate:.2%}")
print(f"Pass@5: {results.pass_at_k[5]:.2%}")
```

## Documentation

### Integration Guides

- **[Installation Guide](docs/installation.md)** - How to install and use the framework
- **[StrideAI Integration](docs/strideai-integration.md)** - Phase 2A: LLM graders for goal decomposition
- **[Crypto-Trading Integration](docs/crypto-trading-integration.md)** - Phase 2B: Financial metrics graders
- **[CI/CD Integration](docs/ci-cd-integration.md)** - Phase 3: Automated evaluation pipelines

### References

- [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

## Key Design Principles

From Anthropic's evaluation guide:

1. **Grade outcomes, not execution paths** - Focus on what the agent produced
2. **Handle non-determinism with pass@k and pass^k** - Different metrics for capability vs reliability
3. **Start with 20-50 real failure cases** - Build from actual issues
4. **Read transcripts regularly** - Catch false signals and grader bugs
5. **Calibrate with human evaluation** - LLM graders drift without calibration
