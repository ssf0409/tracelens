# Multi-Level Evaluation Architecture

How to evaluate AI agents at different levels of granularity using tracelens.

## Overview

Agent evaluation isn't one-size-fits-all. A trading signal calculator needs different evaluation than a full trading pipeline. A goal parser needs different evaluation than an end-to-end goal decomposition agent. tracelens operates at the **Task** level — one Task, one adapter call, one Transcript — but what you put *inside* that Task determines the evaluation granularity.

This document defines three evaluation levels, shows how to implement each using the existing framework, and identifies gaps for future first-class support.

### The Three Levels

| | Function | Task | System |
|---|---|---|---|
| **Analogy** | Unit test | Integration test | End-to-end test |
| **Scope** | Single component | One agent invocation | Multi-step pipeline |
| **What's tested** | Parser, scorer, tool, LLM call | Complete agent on one goal | Chained agents across stages |
| **Typical grader** | `CodeGrader` | `CodeGrader` + `LLMGrader` | `CompositeGrader` |
| **Primary statistic** | pass@1 | pass@k | pass^k |
| **Run count** | 1–3 | 3–5 | 5–10 |
| **Speed** | Fast (ms) | Moderate (seconds) | Slow (minutes) |

---

## Function-Level Evaluation (Component Isolation)

**What you're testing**: A single component in isolation — an LLM call, a tool, a parser, a scoring function. The agent's internal building blocks.

**Why it matters**: If a component is broken, the full agent will fail. Function-level evals catch regressions at the source, before they cascade into confusing end-to-end failures.

### Convention

Use `Task.category = "function"` and `Task.metadata` to identify the component:

```python
from tracelens import Task

# Evaluate the goal parser in isolation
parser_task = Task(
    name="Parse compound fitness goal",
    category="function",
    tags=["function", "parser"],
    metadata={
        "component": "goal_parser",
        "level": "function",
    },
    input_data={
        "raw_input": "I want to run a marathon in under 4 hours and lose 10 pounds",
    },
    expectation=TaskExpectation(
        expected_output={
            "goals": [
                {"type": "fitness", "target": "marathon", "constraint": "under 4 hours"},
                {"type": "health", "target": "weight_loss", "amount": "10 pounds"},
            ]
        }
    ),
)
```

### Adapter

Write a thin adapter that calls the component directly, **not** the full agent:

```python
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript

class GoalParserAdapter(AgentAdapter):
    """Calls the goal parser component directly."""

    async def run(self, task: Task) -> Transcript:
        from my_agent.parsing import parse_goal  # Your component

        transcript = self.start_transcript(task)
        result = parse_goal(task.input_data["raw_input"])
        transcript.final_output = result
        transcript.completed_at = datetime.utcnow()
        return transcript
```

### Grader

`CodeGrader` with deterministic assertions. Function-level evals should have clear right/wrong answers:

```python
from tracelens.core.grader import CodeGrader

class GoalParserGrader(CodeGrader):
    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = task.expectation.expected_output["goals"]
        actual = transcript.final_output.get("goals", [])
        return {
            "goal_count_match": float(len(actual) == len(expected)),
            "types_match": float(
                {g["type"] for g in actual} == {g["type"] for g in expected}
            ),
        }

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        all_match = all(v == 1.0 for v in metrics.values())
        score = sum(metrics.values()) / len(metrics)
        return all_match, score
```

### Statistics

- **Deterministic components** (parsers, calculators): pass@1 is sufficient. Run once.
- **LLM-based components** (an LLM call in isolation): Use pass@k with `num_runs=3` to account for non-determinism.

### Concrete Examples

**Goal-decomposition agent:**
| Component | Input | Expected Output |
|---|---|---|
| Goal parser | Raw user text | Structured goal objects |
| Priority scorer | Goals + user context | Priority-ordered list with scores |
| Time estimator | Task + difficulty | Hours estimate within 20% of reference |

**Algorithmic-trading agent:**
| Component | Input | Expected Output |
|---|---|---|
| Indicator calculator | OHLCV candles | RSI/MACD/Bollinger values matching reference |
| Risk validator | Position + portfolio | Accept/reject with reason |
| Signal classifier | Market features | Buy/sell/hold label |

---

## Task-Level Evaluation (Single Agent Invocation)

**What you're testing**: One complete agent invocation — the current default mode. Feed a task to the full agent, get a result, grade it.

**Why it matters**: This is the bread-and-butter evaluation. It tells you whether the agent can actually solve the problem it was designed for.

### Convention

`Task.category = "task"` (or omit it — this is the default):

```python
task = Task(
    name="Decompose beginner web portfolio goal",
    category="task",
    tags=["task", "web", "beginner"],
    input_data={
        "goal": "Build a personal portfolio website",
        "user_context": {"experience": "beginner", "hours_per_week": 15},
    },
    difficulty="medium",
)
```

### Adapter

Use the full agent adapter — `SimpleAdapter` for simple callables, or a custom `AgentAdapter`:

```python
from tracelens.execution.agent_adapter import SimpleAdapter

async def invoke_agent(input_data: dict) -> dict:
    from my_agent.agent import GoalDecompositionAgent
    agent = GoalDecompositionAgent()
    return await agent.decompose(input_data["goal"], input_data["user_context"])

adapter = SimpleAdapter(invoke_agent)
```

### Grader

Task-level grading often combines objective checks (MUST_PASS) with subjective quality (SCORE_CONTRIBUTOR):

```python
from tracelens.core.grader import CompositeGrader, GraderConfig, GraderRole

# Format validation — must pass or trial fails
format_config = GraderConfig(role=GraderRole.MUST_PASS)
format_grader = FormatValidationGrader("format", config=format_config)

# Quality assessment — contributes to score
quality_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR, weight=0.6)
quality_grader = DecompositionQualityGrader("quality", config=quality_config)

# Personalization — contributes to score
personalization_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR, weight=0.4)
personalization_grader = PersonalizationGrader("personalization", config=personalization_config)

composite = CompositeGrader(
    grader_id="task_composite",
    graders=[
        (format_grader, 0.1),
        (quality_grader, 0.6),
        (personalization_grader, 0.3),
    ],
)
```

The `CompositeGrader` enforces the role semantics: if `format_grader` (MUST_PASS) fails, the trial fails regardless of quality scores.

### Statistics

- **pass@k** for capability: "Can it solve this at least once in k tries?"
- **pass^k** for reliability: "Will it solve this every time?"
- Recommend `num_runs >= 3` for LLM-based agents, `num_runs = 1` for deterministic agents.

```python
from tracelens.execution.runner import RunnerConfig

config = RunnerConfig(
    num_runs=5,          # 5 runs per task for pass@k and pass^k
    max_concurrency=10,
    timeout_seconds=120.0,
)
```

### Concrete Examples

**Goal-decomposition agent:**
- Decompose "Learn to cook Italian food" for a busy professional
- Decompose "Train for a 5K" for someone with a knee injury
- Decompose "Build a SaaS product" for a solo developer

**Algorithmic-trading agent:**
- Execute a single BTC long trade given market conditions
- Size a position given portfolio constraints and risk limits
- Generate a market analysis report for ETH/USDT

---

## System-Level Evaluation (Multi-Step Pipeline)

**What you're testing**: An end-to-end pipeline spanning multiple agents or stages. The full workflow from input to final output, including intermediate handoffs.

**Why it matters**: Components can each pass in isolation but fail when chained together. System-level evals catch integration failures, error propagation, and emergent behavior that only appears at scale.

### Convention

Use `Task.metadata` to describe the pipeline stages:

```python
task = Task(
    name="Full trading pipeline: signal to confirmation",
    category="system",
    tags=["system", "pipeline", "trading"],
    metadata={
        "level": "system",
        "pipeline": ["signal_generator", "risk_checker", "order_executor", "confirmation"],
        "expected_stages": 4,
    },
    input_data={
        "market_data": {"symbol": "BTC/USDT", "timeframe": "1h"},
        "portfolio": {"balance": 10000, "positions": []},
    },
    timeout_seconds=600.0,  # System-level needs more time
)
```

### Adapter

Write a custom `AgentAdapter` that orchestrates the full pipeline and records intermediate outputs in `Transcript.intermediate_outputs`:

```python
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.core.transcript import Transcript, TranscriptStep, StepType

class TradingPipelineAdapter(AgentAdapter):
    """Runs the full signal → risk → order → confirm pipeline."""

    async def run(self, task: Task) -> Transcript:
        transcript = self.start_transcript(task)

        try:
            # Stage 1: Signal generation
            signal = await self.signal_generator.analyze(task.input_data["market_data"])
            transcript.intermediate_outputs.append({
                "stage": "signal_generator",
                "output": signal,
            })
            transcript.add_step(TranscriptStep(
                step_type=StepType.INTERNAL,
                content={"stage": "signal_generator", "result": signal},
            ))

            # Stage 2: Risk check
            risk_result = await self.risk_checker.evaluate(signal, task.input_data["portfolio"])
            transcript.intermediate_outputs.append({
                "stage": "risk_checker",
                "output": risk_result,
            })

            if not risk_result["approved"]:
                transcript.final_output = {"status": "rejected", "reason": risk_result["reason"]}
                return transcript

            # Stage 3: Order execution
            order = await self.order_executor.execute(signal, risk_result)
            transcript.intermediate_outputs.append({
                "stage": "order_executor",
                "output": order,
            })

            # Stage 4: Confirmation
            confirmation = await self.confirmer.verify(order)
            transcript.intermediate_outputs.append({
                "stage": "confirmation",
                "output": confirmation,
            })

            transcript.final_output = {
                "status": "completed",
                "order": order,
                "confirmation": confirmation,
            }
        except Exception as exc:
            self.record_error(transcript, exc)
            raise
        finally:
            transcript.completed_at = datetime.utcnow()

        return transcript
```

### Grader

Use `CompositeGrader` with MUST_PASS gates for pipeline completion and safety, plus SCORE_CONTRIBUTOR for end-to-end quality:

```python
class PipelineCompletionGrader(CodeGrader):
    """MUST_PASS: Did the pipeline complete all expected stages?"""

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = task.metadata.get("expected_stages", 0)
        actual = len(transcript.intermediate_outputs)
        return {
            "stages_completed": float(actual),
            "stages_expected": float(expected),
            "completion_ratio": actual / expected if expected > 0 else 0.0,
        }

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        passed = metrics["completion_ratio"] >= 1.0
        return passed, metrics["completion_ratio"]


class SafetyGateGrader(CodeGrader):
    """MUST_PASS: Were risk limits respected throughout the pipeline?"""

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        # Check risk_checker stage output
        risk_stage = next(
            (o for o in transcript.intermediate_outputs if o["stage"] == "risk_checker"),
            None,
        )
        risk_evaluated = 1.0 if risk_stage is not None else 0.0

        # Check position size limits
        final = transcript.final_output or {}
        order = final.get("order", {})
        position_pct = order.get("position_size_pct", 0)
        within_limits = 1.0 if position_pct <= 5.0 else 0.0  # Max 5% per position

        return {"risk_evaluated": risk_evaluated, "within_limits": within_limits}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        passed = all(v == 1.0 for v in metrics.values())
        return passed, sum(metrics.values()) / len(metrics)
```

Assemble them:

```python
composite = CompositeGrader(
    grader_id="system_composite",
    graders=[
        # Gates — must pass
        (PipelineCompletionGrader("completion", config=GraderConfig(role=GraderRole.MUST_PASS)), 0.1),
        (SafetyGateGrader("safety", config=GraderConfig(role=GraderRole.MUST_PASS)), 0.1),
        # Quality — score contributors
        (EndToEndPnLGrader("pnl", config=GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)), 0.5),
        (ExecutionQualityGrader("exec_quality", config=GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)), 0.3),
    ],
)
```

### Statistics

- **pass^k is critical** — pipeline reliability is the primary concern. A pipeline that works 80% of the time is not production-ready.
- **Bootstrap CI** for confidence on end-to-end metrics.
- Recommend `num_runs >= 5` (preferably 10) for meaningful pass^k estimates.

```python
from tracelens.statistics.consistency import ConsistencyAnalyzer

analyzer = ConsistencyAnalyzer(k_values=[3, 5])
stability = analyzer.compute_stability_metrics(results_per_task)
# stability["pass^3"] = 0.6  → 60% of 3-run windows pass every time
# stability["pass^5"] = 0.2  → only 20% of 5-run windows are fully consistent
# stability["reliability_score"] = weighted combination
```

### Concrete Examples

**Goal-decomposition pipeline:**
1. Goal parsing → decomposition → execution plan → resource selection → validation
2. Test: "I want to transition from accounting to data science in 6 months"
3. Grading: Did it produce a valid multi-phase plan? Is each phase achievable? Are resources appropriate for the user's background?

**Algorithmic-trading pipeline:**
1. Signal generation → risk assessment → order placement → execution confirmation → monitoring setup
2. Test: "BTC/USDT shows bullish divergence on 1h timeframe with $10k portfolio"
3. Grading: Did the pipeline complete? Were risk limits respected? Was the position sized correctly? Did confirmation succeed?

---

## Choosing the Right Level

| You want to know... | Use | Example |
|---|---|---|
| "Is this component producing correct outputs?" | **Function** | Goal parser returns valid structured goals |
| "Can the agent solve this problem?" | **Task** (pass@k) | Agent decomposes a fitness goal into a plan |
| "Is the agent reliable on this problem?" | **Task** (pass^k) | Agent consistently produces good decompositions |
| "Does the full pipeline work end-to-end?" | **System** | Signal → risk → order → confirmation completes |
| "Is the pipeline production-reliable?" | **System** (pass^k) | Pipeline succeeds 95%+ of the time |

### Recommended Suite Composition

Start with this ratio and adjust based on your project maturity:

| Level | % of suite | Rationale |
|---|---|---|
| Function | 50–60% | Fast, cheap, catches regressions early |
| Task | 30–40% | Core capability validation |
| System | 10–20% | Expensive but catches integration issues |

As the project matures and components stabilize, shift weight from function to system.

---

## Mixing Levels in One EvalSet

### Using `Task.category` for Filtering

All three levels can coexist in a single eval set. Use `EvalSet.filter_tasks()` or `EvalSet.filtered_eval_set()` to run subsets:

```python
from tracelens.core.task import EvalSet

# Full suite with mixed levels
full_suite = EvalSet(name="My Agent — Complete", tasks=all_tasks)

# Run only function-level evals (fast, for pre-commit)
function_tasks = full_suite.filter_tasks(categories=["function"])

# Run only task-level evals (medium, for CI)
task_tasks = full_suite.filter_tasks(categories=["task"])

# Run only system-level evals (slow, for nightly)
system_suite = full_suite.filtered_eval_set(categories=["system"])
```

### Multi-Dimensional Filtering with Tags

Tags encode both level and domain, enabling cross-cutting queries:

```python
# All parser-related evals, any level
parser_evals = full_suite.filter_tasks(tags=["parser"])

# All planner function-level evals
planner_functions = full_suite.filter_tasks(
    categories=["function"],
    tags=["planner"],
)
```

### Example `tasks.json` with Mixed Levels

```json
{
  "tasks": [
    {
      "name": "Parse compound goal",
      "category": "function",
      "tags": ["function", "parser"],
      "metadata": {"component": "goal_parser", "level": "function"},
      "input_data": {"raw_input": "Run a marathon and lose weight"}
    },
    {
      "name": "Decompose beginner fitness goal",
      "category": "task",
      "tags": ["task", "fitness"],
      "input_data": {"goal": "Get fit for summer", "user_context": {"experience": "beginner"}}
    },
    {
      "name": "Full decomposition pipeline",
      "category": "system",
      "tags": ["system", "pipeline"],
      "metadata": {"level": "system", "pipeline": ["parser", "decomposer", "validator"]},
      "input_data": {"goal": "Career transition to data science", "user_context": {"background": "accounting"}}
    }
  ]
}
```

### Running Subsets from Code

```python
import asyncio
from tracelens.execution.runner import EvaluationRunner, RunnerConfig

# Different configs per level
level_configs = {
    "function": RunnerConfig(num_runs=1, max_concurrency=20, timeout_seconds=30),
    "task":     RunnerConfig(num_runs=5, max_concurrency=10, timeout_seconds=120),
    "system":   RunnerConfig(num_runs=10, max_concurrency=3, timeout_seconds=600),
}

for level, config in level_configs.items():
    subset = full_suite.filtered_eval_set(categories=[level])
    if not subset.tasks:
        continue

    runner = EvaluationRunner(adapters[level], graders[level], config)
    batch = asyncio.run(runner.run(subset))
    print(f"{level}: pass_rate={batch.pass_rate:.2%}")
```

> **Note**: The CLI does not currently support `--categories` or `--tags` flags. Filtering by level must be done in code via `EvalSet.filter_tasks()` or `EvalSet.filtered_eval_set()`. Adding CLI-level filtering is tracked as a future enhancement.

---

## Statistics by Level

| Level | Primary Stat | Secondary Stat | Recommended `num_runs` | Why |
|---|---|---|---|---|
| Function | pass@1 | pass^k (if non-deterministic) | 1–3 | Deterministic components need 1 run. LLM-wrapped components need 3. |
| Task | pass@k | pass^k | 3–5 | Need enough runs for meaningful capability and reliability estimates. |
| System | pass^k | Bootstrap CI | 5–10 | Pipeline reliability is the primary concern. More runs = tighter confidence. |

### Interpreting Results by Level

**Function-level** — Binary. If pass@1 < 1.0, the component is broken. Fix it.

**Task-level** — Nuanced.
- pass@5 = 0.99 but pass^3 = 0.4 → Agent *can* solve it but is inconsistent. Tune temperature, add retries, or improve prompts.
- pass@5 = 0.3 → Agent can't reliably solve this type of problem. Rethink the approach.

**System-level** — Holistic.
- pass^5 = 0.8 → Pipeline succeeds 80% of the time over 5 consecutive runs. Reasonable for staging.
- pass^5 = 0.95+ → Production-ready reliability.
- Use `ConsistencyAnalyzer.compute_stability_metrics()` for `reliability_score` and `avg_longest_streak`.

---

## Baseline Strategy by Level

Each level needs different regression detection sensitivity.

### Function Level

Tight thresholds. Components should be stable.

```python
from tracelens.baselines.manager import BaselineManager, PromotionPolicy

manager = BaselineManager("baselines/baselines.json")

manager.create_capability_baseline(
    task_id="goal_parser_compound",
    metrics={"goal_count_match": 1.0, "types_match": 1.0},
    promotion_policy=PromotionPolicy(
        min_improvement_relative=0.02,  # 2% — tight
        min_samples=5,
    ),
)
```

- **Threshold**: 2% relative decline triggers regression
- **Promotion**: Fast — auto-promote when deterministic component improves
- **Type**: `CAPABILITY` — track improvements over time

### Task Level

Standard thresholds. Allow for LLM non-determinism.

```python
manager.create_capability_baseline(
    task_id="decompose_fitness_goal",
    metrics={"quality_score": 0.78, "personalization_score": 0.72},
    promotion_policy=PromotionPolicy(
        min_improvement_relative=0.05,   # 5% — standard
        min_samples=10,
        required_confidence=0.95,
    ),
)
```

- **Threshold**: 5–10% relative decline
- **Promotion**: Moderate — require confidence and sample size
- **Type**: `CAPABILITY`

### System Level

Wide thresholds. Pipelines have high variance.

```python
# Safety baseline — never auto-updates
manager.create_canary_baseline(
    task_id="trading_pipeline_safety",
    metrics={"risk_compliance": 1.0, "position_limit_respected": 1.0},
    fingerprint="abc123...",  # Tied to specific config
)

# Performance baseline — can auto-update with wide tolerance
manager.create_capability_baseline(
    task_id="trading_pipeline_performance",
    metrics={"pipeline_completion_rate": 0.85, "avg_latency_ms": 4500},
    promotion_policy=PromotionPolicy(
        min_improvement_relative=0.10,   # 10% — wide
        min_samples=20,
        required_confidence=0.95,
    ),
)
```

- **Safety metrics**: `CANARY` baseline — never auto-update, manual promotion only
- **Performance metrics**: `CAPABILITY` with 10–15% relative threshold
- **Type**: Mix of `CANARY` (safety floors) and `CAPABILITY` (performance tracking)

### Summary Table

| Level | Baseline Type | Relative Threshold | Promotion Speed |
|---|---|---|---|
| Function | CAPABILITY | 2% | Fast (auto) |
| Task | CAPABILITY | 5–10% | Moderate (with confidence) |
| System (safety) | CANARY | 0% (must match) | Manual only |
| System (performance) | CAPABILITY | 10–15% | Slow (high sample count) |

---

## Current Gaps and Future Work

The following would add first-class support for multi-level evaluation. These are **not currently implemented** — they document where the framework could evolve.

### Hierarchical Tasks

**Gap**: Tasks are flat. There's no parent/child relationship between a system-level task and the function-level tasks for its components.

**What it would enable**: Define a system-level task that automatically decomposes into function-level sub-tasks. A failure in the system-level eval could automatically identify which component failed.

### Sub-Transcripts

**Gap**: `Transcript.intermediate_outputs` is a `list[Any]` — unstructured. There's no way to nest a full Transcript inside another Transcript.

**What it would enable**: Each pipeline stage gets its own Transcript with steps, timing, and token counts. The parent Transcript aggregates them. Graders could inspect individual stage transcripts.

### Level-Aware Reporting

**Gap**: Reports don't group results by evaluation level. A markdown report mixes function, task, and system results together.

**What it would enable**: Reports with separate sections per level, level-specific summary statistics, and drill-down from system to task to function failures.

### Pipeline Grading (Intermediate Steps)

**Gap**: Graders operate on the final `Transcript` — there's no built-in way to grade intermediate pipeline outputs independently.

**What it would enable**: Grade each pipeline stage with its own grader. A system-level CompositeGrader could include stage-specific graders alongside end-to-end graders.

### CLI-Level Filtering

**Gap**: The `tracelens run` CLI does not accept `--categories` or `--tags` flags. Level-based filtering must be done in code.

**What it would enable**: `tracelens run --eval-set suite.json --categories function` for pre-commit hooks, `--categories system` for nightly runs.

### Cross-Level Baselines

**Gap**: Baselines are per-task. There's no suite-level regression detection that considers the relationship between levels.

**What it would enable**: "Function-level pass rate dropped 5% AND system-level reliability dropped 20% → likely the same root cause." Cross-level correlation analysis for faster debugging.

---

## Framework API Reference

Features referenced in this document and where to find them:

| Feature | File | Line | How It's Used |
|---|---|---|---|
| `Task.category` | `src/tracelens/core/task.py` | 70 | Encode evaluation level |
| `Task.metadata` | `src/tracelens/core/task.py` | 65 | Store component/pipeline info |
| `Task.tags` | `src/tracelens/core/task.py` | 66 | Multi-dimensional filtering |
| `Task.matches_filter()` | `src/tracelens/core/task.py` | 76–89 | Filter predicate |
| `EvalSet.filter_tasks()` | `src/tracelens/core/task.py` | 195–209 | Get filtered task list |
| `EvalSet.filtered_eval_set()` | `src/tracelens/core/task.py` | 211–230 | Get filtered EvalSet |
| `Transcript.intermediate_outputs` | `src/tracelens/core/transcript.py` | 104 | Record pipeline stages |
| `CompositeGrader` | `src/tracelens/core/grader.py` | 321–447 | Multi-grader aggregation |
| `GraderRole` (MUST_PASS / SCORE_CONTRIBUTOR) | `src/tracelens/core/grader.py` | 30–43 | Gate vs. quality grading |
| `AgentAdapter` ABC | `src/tracelens/execution/agent_adapter.py` | 15–50 | Custom adapters per level |
| `SimpleAdapter` | `src/tracelens/execution/agent_adapter.py` | 53–84 | Wrap callables |
| `RunnerConfig.num_runs` | `src/tracelens/execution/runner.py` | 27 | Level-appropriate run counts |
| `BaselineType` (CANARY / CAPABILITY / EXPERIMENTAL) | `src/tracelens/baselines/manager.py` | 21–44 | Level-specific baseline strategy |
| `PromotionPolicy` | `src/tracelens/baselines/manager.py` | 47–76 | Auto-promotion criteria |
| `RegressionSeverity` | `src/tracelens/baselines/comparison.py` | 18–24 | Regression blocking thresholds |
| `pass_at_k()` | `src/tracelens/statistics/pass_at_k.py` | 15–46 | Capability estimation |
| `pass_to_k()` | `src/tracelens/statistics/consistency.py` | 13–43 | Reliability estimation |
| `ConsistencyAnalyzer` | `src/tracelens/statistics/consistency.py` | 82–191 | Stability metrics |
| `bootstrap_ci()` | `src/tracelens/statistics/inference.py` | 136–194 | Confidence intervals |
