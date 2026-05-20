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

### pass@k — Capability

"What's the probability of at least one success in k attempts?"

```python
from tracelens.statistics.pass_at_k import pass_at_k, PassAtKAnalyzer

# Single task: 10 runs, 7 passed, what's pass@5?
prob = pass_at_k(n=10, c=7, k=5)  # 0.99+

# Multiple tasks with confidence intervals
analyzer = PassAtKAnalyzer(k_values=[1, 3, 5])
results = analyzer.analyze(pass_results_by_task)
# {"pass@1": 0.7, "pass@3": 0.92, "pass@5": 0.98}

results_with_ci = analyzer.analyze_with_ci(pass_results_by_task)
# {"pass@5": {"value": 0.98, "lower": 0.92, "upper": 1.0}}
```

### pass^k — Reliability

"What's the probability that ALL k attempts succeed?"

```python
from tracelens.statistics.consistency import pass_to_k, ConsistencyAnalyzer

# Single task: [T, T, F, T, T], what's pass^3?
prob = pass_to_k([True, True, False, True, True], k=3)  # 0.333

# Multiple tasks
analyzer = ConsistencyAnalyzer(k_values=[2, 3, 5])
results = analyzer.analyze(pass_results_by_task)
# {"pass^2": 0.85, "pass^3": 0.70, "pass^5": 0.40}

# Stability metrics (includes reliability score, failure rate, streaks)
stability = analyzer.compute_stability_metrics(pass_results_by_task)
```

### Bootstrap CI and Comparison

```python
from tracelens.statistics.inference import compare_metrics, estimate_metric

# Compare current run against baseline
result = compare_metrics(
    baseline_values=[0.72, 0.75, 0.71, 0.74],
    current_values=[0.78, 0.81, 0.79, 0.82],
    confidence=0.95,
    compute_p_value=True,
)

print(f"Difference: {result.difference:.3f}")
print(f"95% CI: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
print(f"Effect size: {result.effect_size:.2f}")
print(f"Significant: {result.significant_improvement}")
```

### Which Statistic to Use

| Question | Statistic | When |
|----------|-----------|------|
| Can it solve this? | pass@k | Capability benchmarking |
| Is it reliable? | pass^k | Production readiness |
| Is it better than before? | compare_metrics | A/B testing, regression |
| How confident are we? | bootstrap CI | Any comparison |

## Reproducibility

### DecisionSpec

Captures all parameters affecting agent behavior:

```python
from tracelens.core.decision_spec import (
    DecisionSpec, ModelConfig, AgentSpec, PromptSpec, ToolSpec
)

spec = DecisionSpec(
    model=ModelConfig(
        model_id="gpt-4-turbo",
        temperature=0.7,
        max_tokens=4096,
    ),
    agent=AgentSpec(
        agent_id="goal-decomposer",
        version="1.2.3",
        git_commit="abc123",
    ),
    prompts=[PromptSpec(name="system", version="v2", hash="...")],
    tools=[ToolSpec(name="search", version="1.0")],
    global_seed=42,
)

# SHA-256 fingerprint of entire configuration
print(spec.fingerprint)       # Full hash
print(spec.fingerprint_short) # First 12 chars

# Attach to transcript
transcript.decision_spec = spec
```

### Fingerprint Comparison

When comparing baselines, mismatched fingerprints indicate configuration changes. This helps distinguish "agent got worse" from "we changed the configuration."

## Baselines and Regression

### BaselineManager

```python
from tracelens.baselines import BaselineManager

manager = BaselineManager("baselines.json")

# Get existing baseline
baseline = manager.get_baseline("task-1")

# Update with new results
manager.update_baseline(
    "task-1",
    metrics={"pass_rate": 0.85, "mean_score": 0.78},
    metric_stds={"pass_rate": 0.03, "mean_score": 0.05},
    sample_size=50,
)

# Save to disk
manager.save()
```

### Baseline Types

| Type | Auto-Update | Use For |
|------|-------------|---------|
| `CANARY` | Never | Safety-critical, business-critical metrics |
| `CAPABILITY` | On improvement | Tracking progress, regression detection |
| `EXPERIMENTAL` | Aggressively | Active development, prototyping |

```python
# Canary — never auto-updates, requires fingerprint
manager.create_canary_baseline(
    task_id="safety",
    metrics={"safety_score": 0.99},
    fingerprint=spec.fingerprint,
)

# Capability — auto-promotes on improvement
from tracelens.baselines.manager import PromotionPolicy
manager.create_capability_baseline(
    task_id="quality",
    metrics={"quality_score": 0.75},
    promotion_policy=PromotionPolicy(
        min_improvement_relative=0.05,
        min_samples=10,
    ),
)
```

### PromotionPolicy

Controls when baselines can be auto-updated:

```python
policy = PromotionPolicy(
    allow_auto_promotion=True,
    min_improvement_relative=0.05,  # 5% improvement required
    min_samples=10,                 # Minimum sample size
    required_confidence=0.95,       # 95% confidence
    max_age_days=30,                # Flag stale baselines
    require_all_metrics_improve=False,
    promotion_cooldown_days=7,      # 7-day cooldown between promotions
)
```

### RegressionDetector

```python
from tracelens.baselines.comparison import RegressionDetector, RegressionSeverity

detector = RegressionDetector(significance_level=0.05)
report = detector.compare(baseline, current_results)

# Severity levels: NONE < MINOR < MODERATE < SEVERE
if report.should_block_ci(threshold=RegressionSeverity.MODERATE):
    print(report.to_ci_output())
    sys.exit(1)
```

| Severity | Change | Default CI Action |
|----------|--------|-------------------|
| NONE | No regression | Pass |
| MINOR | < 5% decline | Pass |
| MODERATE | 5-15% decline | Block |
| SEVERE | > 15% decline | Block |

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
