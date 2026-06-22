# TraceLens User Guide

Comprehensive guide to the TraceLens evaluation framework.

## Architecture Overview

TraceLens follows a linear pipeline:

```
Task → Adapter → Transcript → Grader → Outcome → Trial → TrialBatch → Report
```

1. **Task** defines what to evaluate (input data, expected behavior)
2. **AgentAdapter** invokes your agent and captures a **Transcript**
3. **Grader** evaluates the transcript and produces an **Outcome** (pass/fail + score)
4. A **Trial** bundles transcript + outcomes for one task × one run
5. **TrialBatch** collects all trials for statistical analysis
6. **ReportGenerator** produces markdown, CI summaries, and HTML dashboards

## Tasks and EvalSets

### Task Fields

```python
from tracelens import Task

task = Task(
    task_id="unique-id",              # Auto-generated UUID if omitted
    name="Portfolio website plan",     # Required: human-readable name
    input_data={                       # Required: what the agent receives
        "goal": "Build a portfolio website",
        "context": {"experience": "beginner"},
    },
    description="Test goal decomposition for a beginner web project",
    category="programming",            # For filtering
    tags=["web", "beginner"],          # For filtering
    difficulty="medium",               # "easy", "medium", "hard"
    metadata={"expected_steps": 5},    # Grader-specific data
    timeout_seconds=300.0,             # Per-task timeout
    max_retries=1,
)
```

### TaskExpectation

Optional structured expectations for graders that need them:

```python
from tracelens.core.task import TaskExpectation

task = Task(
    name="Format check",
    input_data={"goal": "..."},
    expectation=TaskExpectation(
        expected_output={"format": "json"},
        expected_tool_calls=["search", "write"],
        metric_thresholds={"quality": 0.7},
    ),
)
```

### JSONTaskLoader

Load tasks from JSON files:

```python
from tracelens.core.task import JSONTaskLoader

loader = JSONTaskLoader()

# From a single file (supports {"tasks": [...]} or bare list)
tasks = loader.load("eval/tasks.json")

# From a directory (recursively loads all .json files)
tasks = loader.load("eval/scenarios/")

# Save tasks
loader.save(tasks, "output/tasks.json")
```

### EvalSet and Filtering

An `EvalSet` groups tasks together:

```python
from tracelens import EvalSet

eval_set = EvalSet(
    name="Goal Decomposition v1",
    tasks=tasks,
    default_num_runs=5,
    default_grader_ids=["quality", "personalization"],
)

# Filter tasks
easy_tasks = eval_set.filter_tasks(difficulties=["easy"])
web_tasks = eval_set.filter_tasks(tags=["web"])

# Get a filtered EvalSet (preserves configuration)
subset = eval_set.filtered_eval_set(categories=["programming"], max_tasks=10)
```

## Agent Adapters

### AgentAdapter ABC

The abstract base class for all adapters:

```python
from tracelens import AgentAdapter
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript

class MyAdapter(AgentAdapter):
    async def run(self, task: Task) -> Transcript:
        # 1. Start transcript (sets timing)
        transcript = self.start_transcript(task)

        # 2. Invoke your agent
        try:
            result = await my_agent.invoke(task.input_data)
            transcript.final_output = result
        except Exception as exc:
            self.record_error(transcript, exc)
            raise

        # 3. Complete transcript
        from datetime import datetime
        transcript.completed_at = datetime.utcnow()
        return transcript
```

### SimpleAdapter

Wraps any async callable — ideal for testing and simple agents:

```python
from tracelens import SimpleAdapter

async def my_fn(input_data: dict) -> dict:
    return {"answer": 42}

adapter = SimpleAdapter(my_fn)
```

`SimpleAdapter` handles transcript creation, step recording, error handling, and timing automatically.

### Writing Custom Adapters

For agents with complex invocation patterns:

```python
class LangChainAdapter(AgentAdapter):
    def __init__(self, chain):
        self.chain = chain

    async def run(self, task: Task) -> Transcript:
        transcript = self.start_transcript(task)

        # Record intermediate steps
        for step in await self.chain.astream(task.input_data):
            transcript.add_step(TranscriptStep(
                step_type=StepType.AGENT_OUTPUT,
                content=step,
            ))

        transcript.final_output = step  # Last step is final output
        transcript.completed_at = datetime.utcnow()
        return transcript
```

## Transcripts

A `Transcript` is a complete execution record:

```python
from tracelens.core.transcript import Transcript, TranscriptStep, StepType

transcript = Transcript(task_id="task-1")

# Record steps
transcript.add_step(TranscriptStep(
    step_type=StepType.LLM_CALL,
    model="gpt-4",
    tokens_in=500,
    tokens_out=200,
    content="Planning step output...",
))

transcript.add_step(TranscriptStep(
    step_type=StepType.TOOL_CALL,
    tool_call=ToolCall(tool_name="search", arguments={"q": "python"}),
))

# Access aggregated stats
print(f"Total tokens: {transcript.total_tokens}")
print(f"LLM calls: {transcript.llm_calls_count}")
print(f"Duration: {transcript.duration_ms}ms")
```

Transcripts are invaluable for debugging. When a grader produces unexpected results, read the transcript to understand what the agent actually did.

## Graders

### CodeGrader

Deterministic grading based on computed metrics:

```python
from tracelens import CodeGrader

class AccuracyGrader(CodeGrader):
    def __init__(self):
        super().__init__(grader_id="accuracy")

    def compute_metrics(self, transcript, task):
        """Extract metrics from agent output."""
        output = transcript.final_output
        expected = task.metadata["expected"]
        return {
            "exact_match": float(output == expected),
            "length_ratio": len(str(output)) / max(len(str(expected)), 1),
        }

    def determine_pass(self, metrics, task):
        """Return (passed, score) from metrics."""
        passed = metrics["exact_match"] == 1.0
        score = metrics["exact_match"]
        return passed, score
```

### LLMGrader

LLM-as-judge grading for subjective quality:

```python
from tracelens import LLMGrader

class ClarityGrader(LLMGrader):
    def __init__(self):
        super().__init__(grader_id="clarity", model="gpt-4")

    def build_grading_prompt(self, transcript, task):
        """Build the evaluation prompt."""
        return f"""Score this output on clarity (1-10):
        {transcript.final_output}
        Return JSON: {{"score": N, "feedback": "..."}}"""

    def parse_llm_response(self, response, task):
        """Parse LLM JSON response into (passed, score, metrics, feedback)."""
        import json
        data = json.loads(response)
        score = data["score"] / 10.0
        return score >= 0.7, score, {"clarity": score}, data["feedback"]

    async def _call_llm(self, prompt):
        """Integrate with your LLM client."""
        # Implement with OpenAI, Anthropic, LiteLLM, etc.
        response = await client.chat.completions.create(...)
        return response.choices[0].message.content
```

### CompositeGrader

Combines multiple graders with role-based aggregation:

```python
from tracelens import CompositeGrader, GraderRole, GraderConfig

# Safety grader — must pass or entire trial fails
safety_config = GraderConfig(role=GraderRole.MUST_PASS)
safety = FormatValidator("format", config=safety_config)

# Quality grader — contributes to weighted score
quality_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)
quality = ClarityGrader()

composite = CompositeGrader(
    grader_id="combined",
    graders=[
        (safety, 0.2),    # 20% weight
        (quality, 0.8),   # 80% weight
    ],
)
```

**Aggregation rules:**
- If ANY `MUST_PASS` grader fails → trial fails (regardless of scores)
- Score is a weighted average of ALL graders
- Both roles contribute to the final score

### GraderConfig

```python
config = GraderConfig(
    pass_threshold=0.5,      # Score threshold for passing
    timeout_seconds=60.0,    # Grading timeout
    retry_on_error=True,     # Retry on grading errors
    max_retries=3,
    model="gpt-4",           # LLM model (for LLMGraders)
    temperature=0.0,         # LLM temperature
    weight=1.0,              # Weight in composite scoring
    role=GraderRole.SCORE_CONTRIBUTOR,
)
```

## Outcomes and Scoring

### Outcome Fields

```python
from tracelens.core.outcome import Outcome, GradeLevel

outcome = Outcome(
    trial_id="...",
    grader_id="quality",
    passed=True,
    score=0.85,                     # Normalized 0-1
    metrics={"clarity": 0.9},       # Grader-specific metrics
    grade_level=GradeLevel.GOOD,    # Auto-computed from score
    feedback="Clear and well-organized",
    confidence=0.92,                # For non-deterministic graders
)
```

### GradeLevel

Automatic categorical mapping:

| Score Range | Grade Level |
|-------------|-------------|
| >= 0.9 | EXCELLENT |
| >= 0.7 | GOOD |
| >= 0.5 | ACCEPTABLE |
| >= 0.3 | POOR |
| < 0.3 | FAIL |

### AggregatedOutcome

Suite-level statistics from multiple outcomes:

```python
from tracelens.core.outcome import AggregatedOutcome

agg = AggregatedOutcome.from_outcomes(outcomes)
print(f"Pass rate: {agg.pass_rate:.1%}")
print(f"Mean score: {agg.mean_score:.3f} ± {agg.std_score:.3f}")
print(f"Per-grader pass rates: {agg.grader_pass_rates}")
```

## Trial Execution

### EvaluationRunner

Orchestrates parallel execution with concurrency control:

```python
from tracelens import EvaluationRunner, RunnerConfig

config = RunnerConfig(
    num_runs=5,             # Runs per task (for pass@k)
    max_concurrency=10,     # Max parallel trials
    timeout_seconds=300.0,  # Per-trial timeout
    fail_fast=False,        # Continue on individual failures
)

runner = EvaluationRunner(adapter, [grader1, grader2], config)
batch = await runner.run(eval_set)
```

The runner:
1. Creates `task × num_runs` work items
2. Executes them concurrently (bounded by `max_concurrency`)
3. Enforces per-trial timeouts
4. Grades each trial's transcript with all graders
5. Collects results into a `TrialBatch`

### TrialBatch

Access results from a batch:

```python
# Suite-level stats
print(f"Total: {batch.total_count}")
print(f"Passed: {batch.passed_count}")
print(f"Pass rate: {batch.pass_rate:.1%}")

# Per-task results (for pass@k)
pass_results = batch.get_pass_results_by_task()
# {"task-1": [True, True, False], "task-2": [True, True, True]}

# Individual trials
for trial in batch.get_trials_for_task("task-1"):
    print(f"  Run {trial.run_index}: {trial.passed} (score={trial.aggregate_score})")
```

## Statistical Analysis

Two questions, two metrics: **pass@k** (capability — "can it solve this at all in
k tries?") and **pass^k** (reliability — "does it succeed every time?"). They move
in opposite directions as k grows.

```python
from tracelens import pass_at_k, pass_to_k

pass_at_k(n=10, c=7, k=5)                        # capability: >=1 of 5 passes
pass_to_k([True, True, False, True, True], k=3)  # reliability: all 3 in a window
```

`ReportGenerator(k_values=..., consistency_k_values=...)` computes both for a
batch. To decide whether a difference between two runs is *real* (bootstrap CI,
effect size, p-value), use `compare_metrics`. Where to go deep:

- Intuition + truth table: [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md).
- CIs, effect size, significance, sample size: [Statistical Comparison](statistical-comparison.md).
- Applied A/B across model/prompt versions: [Comparing Versions](comparing-versions.md).

## Reproducibility

Every run can carry a `DecisionSpec` — a content fingerprint of the model,
prompt, tools, agent, and infra that produced it. Pass it to the runner and it
stamps every transcript, so a baseline records the exact config behind it and a
regression becomes attributable to the agent vs. the environment.

```python
runner = EvaluationRunner(adapter, graders, config, decision_spec=spec)
```

The full field reference (and the rule for what enters the fingerprint vs. what's
left out) is in [Reproducibility & DecisionSpec](reproducibility.md).

## Baselines and Regression

Store a known-good result as a baseline, then compare future runs against it and
gate CI on the severity of any decline (NONE < MINOR < MODERATE < SEVERE).
Baseline types encode how cautious promotion should be:

| Type | Auto-update | Use for |
|------|-------------|---------|
| `CANARY` | Never (manual only) | Safety/business-critical floors |
| `CAPABILITY` | On significant improvement | Tracking progress, regression detection |
| `EXPERIMENTAL` | Loosely | Active development, prototyping |

The complete store → promote → compare → gate workflow, with the real
`BaselineManager` and `RegressionDetector` APIs, is the
[Baseline Regression Tutorial](baseline-regression-tutorial.md).

## Reporting

### ReportGenerator

```python
from tracelens.reporting.generator import ReportGenerator

gen = ReportGenerator(
    k_values=[1, 3, 5],               # pass@k values to compute
    consistency_k_values=[2, 3, 5],    # pass^k values to compute
)

report = gen.build_report(batch)

# Output formats
markdown = gen.render_markdown(report)    # Human-readable
ci_line = gen.render_ci_summary(report)   # Compact CI output
html = gen.render_html(report)            # Visual dashboard
```

### ReportData Serialization

Save and reload reports:

```python
import json

# Save
data = report.to_dict()
with open("results.json", "w") as f:
    json.dump(data, f, indent=2, default=str)

# Load
from tracelens.reporting.generator import ReportData
with open("results.json") as f:
    report = ReportData.from_dict(json.load(f))
```

### HTML Dashboard

The HTML report is a self-contained file with inline CSS and SVG charts:

```python
html = gen.render_html(report)
with open("report.html", "w") as f:
    f.write(html)
```

Sections: summary cards, pass@k/pass^k bar charts, per-task results table, pass rate distribution, score distribution histogram, and optional regression alerts.

## CLI

### tracelens run

```bash
tracelens run \
  --eval-set tasks.json \
  --adapter myproject.adapters.MyAdapter \
  --graders myproject.graders.Grader1 myproject.graders.Grader2 \
  --num-runs 5 \
  --max-concurrency 10 \
  --timeout 300 \
  --output results.json \
  --report report.md \
  --html-report report.html \
  --baseline-check \
  --baselines-file baselines.json \
  --fail-on-regression moderate
```

### tracelens report

```bash
# From saved results
tracelens report --results results.json --format markdown
tracelens report --results results.json --format json
tracelens report --results results.json --format html
```

## Plugin System

The `registry` module loads classes from dotted import paths at runtime:

```python
from tracelens.execution.registry import load_class, instantiate

# Load a class
cls = load_class("myproject.graders.QualityGrader")
grader = cls()

# Load and instantiate
grader = instantiate("myproject.graders.QualityGrader", grader_id="quality")
```

This is how the CLI resolves `--adapter` and `--graders` arguments.
