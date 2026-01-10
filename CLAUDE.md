# Agent Evaluation Framework - Development Guide

## Project Overview

This is a **common evaluation framework** for AI agents, designed to be used by:
- **StrideAI** (`/Users/ssf0409/Dev/StrideAI`) - Goal decomposition agent
- **crypto-trading-system** (`/Users/ssf0409/Dev/crypto-trading-system`) - Trading agent

## Architecture Decision

**Hybrid Approach**: Common framework with project-specific layers.

### Common (this package)
- Abstract interfaces (Task, Trial, Grader, Transcript, Outcome)
- Statistical analysis (pass@k, pass^k)
- Baseline management and regression detection
- CI/CD integration patterns
- Human evaluation workflow

### Project-Specific (in each project)
- Task definitions (domain-specific schemas)
- Grader implementations (LLM or code-based)
- Agent adapters (how to invoke the agent)
- Baseline values (project's own regression thresholds)

## Key Files

```
src/agent_eval/
├── core/
│   ├── task.py          # Task, TaskLoader, EvalSet - test case definitions
│   ├── trial.py         # Trial - single execution of a task
│   ├── grader.py        # Grader ABCs - CodeGrader, LLMGrader
│   ├── transcript.py    # Transcript - execution record
│   └── outcome.py       # Outcome - grading result
├── execution/
│   ├── runner.py        # EvaluationRunner - parallel execution
│   └── agent_adapter.py # AgentAdapter ABC - invoke agents
├── statistics/
│   ├── pass_at_k.py     # pass@k - capability ceiling
│   └── consistency.py   # pass^k - reliability measurement
├── baselines/
│   ├── manager.py       # BaselineManager - store/retrieve baselines
│   └── comparison.py    # RegressionDetector - detect regressions
├── human_eval/
│   ├── sampler.py       # HumanEvalSampler - select calibration samples
│   └── reconciliation.py # Reconciler - LLM vs human comparison
└── reporting/
    └── ci_output.py     # CI-friendly output formats
```

## Grader Types

### CodeGrader (for crypto-trading)
Deterministic, code-based grading. Produces the same result every time.

```python
class FinancialGrader(CodeGrader):
    def compute_metrics(self, transcript, task) -> dict[str, float]:
        # Compute metrics from agent output
        return {"sharpe_ratio": 1.5, "max_drawdown": -0.12}

    def determine_pass(self, metrics, task) -> tuple[bool, float]:
        # Determine if task passed based on metrics
        return metrics["sharpe_ratio"] >= 1.0, metrics["sharpe_ratio"] / 2.0
```

### LLMGrader (for StrideAI)
LLM-as-judge grading. Non-deterministic, requires prompt engineering.

```python
class QualityGrader(LLMGrader):
    def build_grading_prompt(self, transcript, task) -> str:
        # Build the prompt for LLM evaluation
        return f"Evaluate quality of: {transcript.final_output}"

    def parse_llm_response(self, response, task):
        # Parse LLM response into structured result
        return passed, score, metrics, feedback
```

## Statistical Analysis

### pass@k (Capability)
"What's the probability of at least one success in k attempts?"

Use for: capability evaluation, "can it do this at all?"

```python
pass_at_k(n=10, c=7, k=5)  # Very high - at least 1 of 5 will likely pass
```

### pass^k (Reliability)
"What's the probability that ALL k attempts succeed?"

Use for: reliability evaluation, "is it consistent?"

```python
pass_to_k(results=[T, T, F, T, T], k=3)  # Lower - must pass every time
```

## Baseline Regression Detection

### Severity Levels
- **NONE**: No regression
- **MINOR**: <5% decline
- **MODERATE**: 5-15% decline (blocks CI by default)
- **SEVERE**: >15% decline

### Threshold Configuration
```python
THRESHOLDS = {
    "sharpe_ratio": {"absolute": -0.2, "relative": 0.10},
    "max_drawdown": {"absolute": -0.05, "relative": 0.20},
}
```

## Integration with Projects

### StrideAI Integration Path
1. `StrideAI/eval/schemas/task_definition.py` - GoalDecompositionTask
2. `StrideAI/eval/graders/*.py` - LLM-based quality graders
3. `StrideAI/eval/data/scenarios/*.json` - 20+ test scenarios
4. `StrideAI/.github/workflows/agent-eval.yml` - CI integration

### Crypto-Trading Integration Path
1. `evaluation/framework/task.py` - TradingTask schema
2. `evaluation/graders/*.py` - Wrap existing metrics
3. `evaluation/baselines/baselines.json` - Regression baselines
4. `evaluation/ci/runner.py` - CI runner

## Testing

```bash
# Run tests
pytest tests/

# Type checking
mypy src/agent_eval/

# Linting
ruff check src/
```

## CI/CD Usage

```bash
# Run evaluation with baseline check
agent-eval run \
  --eval-set eval/suite.json \
  --graders quality,personalization \
  --num-runs 5 \
  --baseline-check \
  --fail-on-regression moderate

# Generate report
agent-eval report --format json --output results.json
```

## Human Evaluation Workflow

Weekly calibration (20 samples):
1. `agent-eval sample --strategy diverse --size 20`
2. Human rates samples in UI
3. `agent-eval reconcile --human human_grades.json --llm llm_grades.json`
4. Review correlation report, adjust graders if < 0.7

## Key Principles (from Anthropic)

1. **Grade outcomes, not paths** - Focus on what agent produced
2. **Handle non-determinism** - Use pass@k for capability, pass^k for reliability
3. **Start with real failures** - Build suite from actual issues
4. **Read transcripts** - Catch false signals and grader bugs
5. **Calibrate regularly** - LLM graders drift without human calibration
