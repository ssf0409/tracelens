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
│   ├── grader.py            # Grader ABC hierarchy (CodeGrader, LLMGrader)
│   ├── transcript.py        # Agent execution logging
│   └── outcome.py           # Grading results
├── execution/               # Trial runner
│   ├── runner.py            # Parallel/concurrent execution
│   └── agent_adapter.py     # Agent invocation interface
├── statistics/              # Non-determinism handling
│   ├── pass_at_k.py         # Capability ceiling (pass@k)
│   └── consistency.py       # Reliability (pass^k)
├── baselines/               # Regression detection
│   ├── manager.py           # Baseline storage
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

```bash
# Basic installation
pip install agent-eval

# With LLM support
pip install agent-eval[llm]

# Development
pip install agent-eval[dev]
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

## Related Documentation

- [Anthropic: Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
- [StrideAI Evaluation Guide](../StrideAI/eval/README.md)
- [Crypto-Trading Evaluation Guide](../crypto-trading-system/docs/evaluation.md)

## Key Design Principles

From Anthropic's evaluation guide:

1. **Grade outcomes, not execution paths** - Focus on what the agent produced
2. **Handle non-determinism with pass@k and pass^k** - Different metrics for capability vs reliability
3. **Start with 20-50 real failure cases** - Build from actual issues
4. **Read transcripts regularly** - Catch false signals and grader bugs
5. **Calibrate with human evaluation** - LLM graders drift without calibration
