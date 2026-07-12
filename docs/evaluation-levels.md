# Multi-Level Evaluation Architecture

How to evaluate AI agents at different levels of granularity using tracelens.

## Overview

Agent evaluation isn't one-size-fits-all. A scoring component needs different
evaluation than a full multi-step pipeline. A parser needs different evaluation
than an end-to-end planning agent. tracelens operates at the **Task** level —
one Task, one adapter call, one Transcript — but what you put *inside* that
Task determines the evaluation granularity.

This document defines three evaluation levels and shows how to implement each —
pick the level that matches the question you're asking, or mix them in one suite.

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

# Evaluate a planning parser in isolation
parser_task = Task(
    name="Parse compound fitness goal",
    category="function",
    tags=["function", "parser", "planning"],
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
    """Calls one parser component directly."""

    async def run(self, task: Task) -> Transcript:
        from myproject.planning import parse_goal  # Your component

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

**Planning agent:**
| Component | Input | Expected Output |
|---|---|---|
| Goal parser | Raw user text | Structured goal objects |
| Priority scorer | Goals + user context | Priority-ordered list with scores |
| Time estimator | Task + difficulty | Hours estimate within 20% of reference |

**Decision agent:**
| Component | Input | Expected Output |
|---|---|---|
| Feature calculator | Raw events | Normalized features matching reference |
| Policy validator | Proposed action + constraints | Accept/reject with reason |
| Action classifier | Context features | Route/hold/escalate label |

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
    tags=["task", "web", "beginner", "planning"],
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

async def invoke_planning_agent(input_data: dict) -> dict:
    from myproject.agent import PlanningAgent
    agent = PlanningAgent()
    return await agent.decompose(input_data["goal"], input_data["user_context"])

adapter = SimpleAdapter(invoke_planning_agent)
```

### Grader

Task-level grading often combines an objective gate (`EvalPolicy.GATE`) with
subjective quality contributors (`EvalPolicy.TRACK`):

```python
from tracelens import CompositeGrader, GraderConfig, EvalPolicy

# Format validation — a hard gate: any violation fails the trial
format_grader = FormatValidationGrader("format", config=GraderConfig(policy=EvalPolicy.GATE))

# Quality + personalization — contribute to the weighted score
quality_grader = DecompositionQualityGrader("quality", config=GraderConfig(policy=EvalPolicy.TRACK))
personalization_grader = PersonalizationGrader("personalization", config=GraderConfig(policy=EvalPolicy.TRACK))

composite = CompositeGrader(
    grader_id="task_composite",
    graders=[
        (format_grader, 0.1),
        (quality_grader, 0.6),
        (personalization_grader, 0.3),
    ],
)
```

If the `GATE` grader fails, the trial fails regardless of the quality scores.
(The older `GraderRole.MUST_PASS` / `SCORE_CONTRIBUTOR` vocabulary still works for
back-compat, but `EvalPolicy` is the current API.)

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

**Decision-support agent:**
- Choose a support escalation path given customer history
- Size a resource allocation given account constraints
- Generate a risk summary from structured event data

---

## System-Level Evaluation (Multi-Step Pipeline)

**What you're testing**: An end-to-end pipeline spanning multiple agents or stages. The full workflow from input to final output, including intermediate handoffs.

**Why it matters**: Components can each pass in isolation but fail when chained together. System-level evals catch integration failures, error propagation, and emergent behavior that only appears at scale.

### Convention

Use `Task.metadata` to describe the pipeline stages:

```python
task = Task(
    name="Full request pipeline: intake to confirmation",
    category="system",
    tags=["system", "pipeline", "operations"],
    metadata={
        "level": "system",
        "pipeline": ["request_parser", "policy_checker", "action_executor", "confirmation"],
        "expected_stages": 4,
    },
    input_data={
        "request": {"kind": "upgrade", "priority": "high"},
        "account": {"tier": "team", "open_cases": 1},
    },
    timeout_seconds=600.0,  # System-level needs more time
)
```

### Adapter

Write a custom `AgentAdapter` that orchestrates the full pipeline and records intermediate outputs in `Transcript.intermediate_outputs`:

```python
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.core.transcript import Transcript, TranscriptStep, StepType

class RequestPipelineAdapter(AgentAdapter):
    """Runs the full parse → policy → execute → confirm pipeline."""

    async def run(self, task: Task) -> Transcript:
        transcript = self.start_transcript(task)

        try:
            # Stage 1: Request parsing
            parsed = await self.request_parser.parse(task.input_data["request"])
            transcript.intermediate_outputs.append({
                "stage": "request_parser",
                "output": parsed,
            })
            transcript.add_step(TranscriptStep(
                step_type=StepType.INTERNAL,
                content={"stage": "request_parser", "result": parsed},
            ))

            # Stage 2: Policy check
            policy_result = await self.policy_checker.evaluate(parsed, task.input_data["account"])
            transcript.intermediate_outputs.append({
                "stage": "policy_checker",
                "output": policy_result,
            })

            if not policy_result["approved"]:
                transcript.final_output = {"status": "rejected", "reason": policy_result["reason"]}
                return transcript

            # Stage 3: Action execution
            action = await self.action_executor.execute(parsed, policy_result)
            transcript.intermediate_outputs.append({
                "stage": "action_executor",
                "output": action,
            })

            # Stage 4: Confirmation
            confirmation = await self.confirmer.verify(action)
            transcript.intermediate_outputs.append({
                "stage": "confirmation",
                "output": confirmation,
            })

            transcript.final_output = {
                "status": "completed",
                "action": action,
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

Use `CompositeGrader` with `EvalPolicy.GATE` for pipeline completion and safety, plus `EvalPolicy.TRACK` for end-to-end quality:

```python
class PipelineCompletionGrader(CodeGrader):
    """GATE: Did the pipeline complete all expected stages?"""

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


class PolicyGateGrader(CodeGrader):
    """GATE: Were project policy constraints respected throughout the pipeline?"""

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        # Check policy_checker stage output
        policy_stage = next(
            (o for o in transcript.intermediate_outputs if o["stage"] == "policy_checker"),
            None,
        )
        policy_evaluated = 1.0 if policy_stage is not None else 0.0

        # Check action limits
        final = transcript.final_output or {}
        action = final.get("action", {})
        estimated_cost = action.get("estimated_cost", 0)
        within_limits = 1.0 if estimated_cost <= 100.0 else 0.0

        return {"policy_evaluated": policy_evaluated, "within_limits": within_limits}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        passed = all(v == 1.0 for v in metrics.values())
        return passed, sum(metrics.values()) / len(metrics)
```

Assemble them:

```python
from tracelens import EvalPolicy, GraderConfig

composite = CompositeGrader(
    grader_id="system_composite",
    graders=[
        # Gates — any failure fails the trial
        (PipelineCompletionGrader("completion", config=GraderConfig(policy=EvalPolicy.GATE)), 0.1),
        (PolicyGateGrader("policy", config=GraderConfig(policy=EvalPolicy.GATE)), 0.1),
        # Quality — score contributors
        (EndToEndQualityGrader("quality", config=GraderConfig(policy=EvalPolicy.TRACK)), 0.5),
        (ExecutionQualityGrader("exec_quality", config=GraderConfig(policy=EvalPolicy.TRACK)), 0.3),
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

**Operations pipeline:**
1. Request parsing → policy assessment → action execution → confirmation → monitoring setup
2. Test: "High-priority upgrade request for a team account with one open case"
3. Grading: Did the pipeline complete? Were policy limits respected? Was the action appropriate? Did confirmation succeed?

---

## Choosing the Right Level

| You want to know... | Use | Example |
|---|---|---|
| "Is this component producing correct outputs?" | **Function** | Goal parser returns valid structured goals |
| "Can the agent solve this problem?" | **Task** (pass@k) | Agent decomposes a fitness goal into a plan |
| "Is the agent reliable on this problem?" | **Task** (pass^k) | Agent consistently produces good decompositions |
| "Does the full pipeline work end-to-end?" | **System** | Parse → policy → execute → confirmation completes |
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
      "tags": ["function", "parser", "planning"],
      "metadata": {"component": "goal_parser", "level": "function"},
      "input_data": {"raw_input": "Run a marathon and lose weight"}
    },
    {
      "name": "Decompose beginner fitness goal",
      "category": "task",
      "tags": ["task", "fitness", "planning"],
      "input_data": {"goal": "Get fit for summer", "user_context": {"experience": "beginner"}}
    },
    {
      "name": "Full decomposition pipeline",
      "category": "system",
      "tags": ["system", "pipeline", "planning"],
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
    task_id="request_pipeline_policy",
    metrics={"policy_compliance": 1.0, "action_limit_respected": 1.0},
    fingerprint="abc123...",  # Tied to specific config
)
# fingerprint is optional when decision_spec= is passed: it is derived from
# the spec, which ties the canary to the actual recording configuration and
# enables noise-aware regression comparison.

# Performance baseline — can auto-update with wide tolerance
manager.create_capability_baseline(
    task_id="request_pipeline_performance",
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
## Where to go deeper

The mechanics referenced above each have a focused home:

- **Every symbol used here** — the [API Reference](reference.md) has full,
  always-current signatures (no line numbers to go stale).
- **Choosing statistics per level** — [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md)
  for the intuition, [Statistical Comparison](statistical-comparison.md) for CIs
  and significance.
- **Baselines per level** — [Baseline Regression Tutorial](baseline-regression-tutorial.md).
- **Attributing a result to a config** — [Reproducibility & DecisionSpec](reproducibility.md).
- **The built-in graders** each level reaches for — [Grader Library](grader-library.md).
