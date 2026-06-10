# TraceLens - Development Guide

## Project Overview

TraceLens is an **open-source evaluation framework for AI agents**. It provides
the project-agnostic machinery — task definitions, trial execution, grading,
statistics, baselines, and human calibration — while consuming projects supply
their own domain layer.

### This package provides
- Abstract interfaces (Task, Trial, Grader, Transcript, Outcome)
- Parallel trial execution with timeout, progress, and checkpoint/resume
- Statistical analysis (pass@k, pass^k, bootstrap CI, significance tests)
- Baseline management and regression detection (CANARY/CAPABILITY, infra-noise aware)
- Reproducibility fingerprinting (DecisionSpec)
- Human evaluation / grader calibration workflow
- CLI: `run` / `report` / `sample` / `calibrate` (alias: `reconcile`)

### Consuming projects supply
- Task definitions (domain-specific schemas)
- Grader implementations (LLM or code-based)
- Agent adapters (how to invoke the agent)
- Baseline values (their own regression thresholds)

See `examples/` for runnable end-to-end integrations (start with
`examples/hello_world.py`) and `benchmarks/` for a complete suite.

## Key Files

```
src/tracelens/
├── core/
│   ├── task.py          # Task, TaskLoader, EvalSet - test case definitions
│   ├── trial.py         # Trial, TrialBatch - execution tracking
│   ├── grader.py        # Grader ABCs - CodeGrader, LLMGrader, CompositeGrader
│   ├── transcript.py    # Transcript - execution record (steps, tokens, streaming)
│   ├── decision_spec.py # DecisionSpec - reproducibility fingerprinting
│   └── outcome.py       # Outcome - grading result (incl. grader_error flag)
├── execution/
│   ├── runner.py        # EvaluationRunner - parallel execution, checkpoint/resume
│   ├── agent_adapter.py # AgentAdapter ABC, SimpleAdapter
│   ├── http_adapter.py  # HTTPAPIAdapter - evaluate agents behind an HTTP API
│   └── registry.py      # Plugin loading via dotted import paths
├── graders/
│   └── event_chain.py   # EventChainGrader - verify behavioral side effects
├── llm/
│   ├── provider.py      # LLMProvider ABC, InMemoryProvider (for tests)
│   └── factory.py       # create_provider() - provider lookup
├── metrics/
│   ├── budgets.py       # Token/latency budget grading
│   └── validators.py    # Schema/constraint validation grading
├── contracts/
│   └── contract.py      # Declarative eval contracts
├── statistics/
│   ├── pass_at_k.py     # pass@k - capability ceiling
│   ├── consistency.py   # pass^k - reliability measurement
│   ├── latency.py       # Latency distribution analysis
│   └── inference.py     # Bootstrap CI, significance testing
├── baselines/
│   ├── manager.py       # BaselineManager - CANARY/CAPABILITY baselines, promotion
│   └── comparison.py    # RegressionDetector - t-test based regression detection
├── reporting/
│   └── generator.py     # ReportGenerator - markdown, CI summary, HTML
├── calibration/
│   ├── analyzer.py      # CalibrationAnalyzer - grader vs human agreement
│   └── sampler.py       # sample_for_review - select trials for human review
└── cli/
    └── main.py          # run / report / sample / calibrate (alias: reconcile)
```

## Grader Types

### CodeGrader
Deterministic, code-based grading. Produces the same result every time.
Use for objective metrics (accuracy, latency, domain KPIs).

```python
class AccuracyGrader(CodeGrader):
    def compute_metrics(self, transcript, task) -> dict[str, float]:
        # Compute metrics from agent output
        return {"accuracy": score_output(transcript.final_output, task)}

    def determine_pass(self, metrics, task) -> tuple[bool, float]:
        # Determine if task passed based on metrics
        return metrics["accuracy"] >= 0.9, metrics["accuracy"]
```

### LLMGrader
LLM-as-judge grading. Non-deterministic, requires prompt engineering and
periodic human calibration. `GraderConfig` controls per-attempt timeout
(`timeout_seconds`) and retry on transient/malformed responses
(`retry_on_error`, `max_retries`, `retry_backoff_seconds`).

```python
class QualityGrader(LLMGrader):
    def build_grading_prompt(self, transcript, task) -> str:
        # Build the prompt for LLM evaluation
        return f"Evaluate quality of: {transcript.final_output}"

    def parse_llm_response(self, response, task):
        # Parse LLM response into structured result
        return passed, score, metrics, feedback
```

### CompositeGrader
Combines graders with policy-aware aggregation (`EvalPolicy.GATE` /
`WARN` / `TRACK`). A sub-grader crash is recorded as a `grader_error`
outcome — surfaced separately from agent failures in batch stats and
reports (`grader_error_count` / `grader_error_rate`).

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

Regressions within the infra-noise band (default 3pp absolute) under a
mismatched infra config don't block CI — see `RegressionDetector` and
`DecisionSpec.infra`.

### Reproducibility
Pass a `DecisionSpec` to `EvaluationRunner` and it is stamped onto every
transcript, so baselines carry a fingerprint of the exact configuration
that produced them.

## Testing & Verification

```bash
# Single-entry verification gate (lint -> typecheck -> tests + coverage)
make verify

# Individual steps
make test        # uv run --frozen pytest -q
make lint        # ruff check
make typecheck   # mypy src/tracelens/
make coverage    # pytest with the 90% coverage floor CI enforces
```

## CI/CD Usage

```bash
# Run evaluation with baseline check
tracelens run \
  --eval-set eval/suite.json \
  --adapter my_pkg.MyAdapter \
  --graders my_pkg.QualityGrader \
  --num-runs 5 \
  --baseline-check \
  --baselines-file baselines.json \
  --fail-on-regression moderate

# Long runs: progress + crash-safe resume
tracelens run ... --progress --checkpoint .tracelens/checkpoint.json

# Generate report
tracelens report --results results.json --format markdown
```

## Human Evaluation Workflow

Periodic calibration to catch LLM-grader drift (see [docs/human-eval.md](docs/human-eval.md)):
1. `tracelens run ... --save-trials trials.json` — keep raw trials.
2. `tracelens sample --trials trials.json --size 20 --strategy diverse --output review.json` —
   select trials to hand-grade. Strategies: `diverse` (span the score range),
   `boundary` (cases nearest the pass/fail line), `failures`, `random`.
3. A human fills `human_score` / `human_passed` in `review.json` (bring your own grades; no UI shipped).
4. `tracelens reconcile --annotations review.json` — the worksheet carries the grader outcome
   next to each human grade, so it pairs them per-row, reports correlation/agreement, and exits
   non-zero below threshold. (`reconcile` is an alias for `calibrate`.)

## Key Principles (from Anthropic)

1. **Grade outcomes, not paths** - Focus on what agent produced
2. **Handle non-determinism** - Use pass@k for capability, pass^k for reliability
3. **Start with real failures** - Build suite from actual issues
4. **Read transcripts** - Catch false signals and grader bugs
5. **Calibrate regularly** - LLM graders drift without human calibration
6. **Separate harness failures from agent failures** - Track infra_error and
   grader_error rates alongside pass rates; a spike there means the eval is
   broken, not the agent
