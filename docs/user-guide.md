# User Guide

You've seen the [pipeline](concepts.md) and run [hello-world](getting-started.md).
Every real eval now comes down to **four decisions**:

1. **What is a task?** — how to scope the unit you evaluate.
2. **How do I invoke my agent?** — which adapter.
3. **How do I grade the outcome?** — which grader(s).
4. **How do I run it and read the results?** — the runner and the statistics.

This guide walks each decision with its trade-offs and a small real example, then
points to the deep page. It is *decision-oriented* — for the exhaustive,
always-current signature of any class, see the [API Reference](reference.md); for
the object model, see [Core Concepts](concepts.md).

| Decision | You choose between | Deep dive |
|----------|--------------------|-----------|
| 1. Task scope | function / task / system level | [Multi-Level Evaluation](evaluation-levels.md) |
| 2. Adapter | `SimpleAdapter` / `HTTPAPIAdapter` / custom | [Evaluating a Real Agent](real-agent.md) |
| 3. Grader | `CodeGrader` / `LLMGrader` / contract / built-ins | [Grader Library](grader-library.md) |
| 4. Run & read | run counts, pass@k vs pass^k, baselines | [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) |

---

## Decision 1 — Define the task

A `Task` is one unit of evaluation: **one task → one adapter call → one
transcript**. Its `input_data` is what gets sent to your agent; `metadata`,
`expectation`, `tags`, and `category` carry the answer key and labels your grader
and filters read (TraceLens itself doesn't interpret them).

```python
from tracelens import Task, EvalSet

eval_set = EvalSet(name="support-suite", tasks=[
    Task(
        name="refund within policy",
        input_data={"ticket": "I want a refund for order #5512"},
        metadata={"expected_action": "refund"},   # your grader reads this
        category="task",
        tags=["billing", "refund"],
    ),
])
```

**Inline vs. from JSON.** Small suites can be inline; real suites live in a
versioned file loaded with `JSONTaskLoader` (the shape is a `{"tasks": [...]}`
envelope):

```python
from pathlib import Path
from tracelens.core.task import JSONTaskLoader

tasks = JSONTaskLoader().load(Path("eval/tasks.json"))
eval_set = EvalSet(name="support-suite", tasks=tasks)
```

**Scope is a real choice.** A task can isolate one component (a parser), one full
agent invocation, or a whole multi-step pipeline. Tag it with `category` and you
can run a fast subset in pre-commit and the full suite in CI:

```python
# Fast pre-commit run: function-level tasks only
fast = eval_set.filtered_eval_set(categories=["function"])
```

→ When to use each scope: [Multi-Level Evaluation](evaluation-levels.md).

---

## Decision 2 — Invoke your agent (the adapter)

The adapter is the only TraceLens code that knows how to *call* your agent. Pick
by how your agent is exposed:

| Your agent is… | Use | Notes |
|----------------|-----|-------|
| an async (or sync) Python callable | `SimpleAdapter(fn)` | Fastest path; `fn(input_data) -> dict`. |
| an HTTP/JSON service | `HTTPAPIAdapter(HTTPAdapterConfig(...))` | Auth, retries, timeout built in. |
| anything else (SDK, multi-step, streaming) | a custom `AgentAdapter` subclass | Implement `async def run(self, task) -> Transcript`. |

```python
from tracelens import SimpleAdapter

async def my_agent(input_data: dict) -> dict:
    return {"action": decide(input_data["ticket"])}

adapter = SimpleAdapter(my_agent)
```

For a custom adapter, call `self.start_transcript(task)` to get a transcript with
timing already started, fill `final_output` (and optionally record steps), and
return it. The runner only depends on the `AgentAdapter` interface, so everything
downstream is identical regardless of which adapter you pick.

→ A custom HTTP adapter end to end: [Evaluating a Real Agent](real-agent.md).

---

## Decision 3 — Grade the outcome

This is where most of the design effort goes. Work down this tree:

- **Is "correct" measurable from the output?** (exact answer, a metric, a schema)
  → **`CodeGrader`**. Deterministic and reproducible.
- **Is it a subjective quality?** (helpfulness, reasoning, tone)
  → **`LLMGrader`** (LLM-as-judge). Non-deterministic — [calibrate it](human-eval.md).
- **Are the rules declarative?** (must include X, must match schema, must not say Y)
  → **`BehaviorContract.to_graders()`** generates the grader suite for you.
- **Is it a common check?** (JSON schema, regex, latency, token budget, tool use,
  event ordering) → it's already in the **[Grader Library](grader-library.md)** —
  don't hand-roll it.

A `CodeGrader` implements two methods — compute metrics, then turn them into a
pass/score:

```python
from tracelens import CodeGrader

class ActionGrader(CodeGrader):
    def compute_metrics(self, transcript, task) -> dict[str, float]:
        got = transcript.final_output.get("action")
        return {"correct": float(got == task.metadata["expected_action"])}

    def determine_pass(self, metrics, task) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]
```

**Combining graders.** Real grading is often a hard gate *plus* a quality score.
`CompositeGrader` takes `(grader, weight)` pairs; each grader's `EvalPolicy`
decides whether it can fail the trial:

- `GATE` — any violation fails the trial (safety, schema).
- `WARN` — recorded, configurably non-blocking.
- `TRACK` — pure signal, contributes to the score only.

```python
from tracelens import CompositeGrader, JsonSchemaGrader

composite = CompositeGrader(
    grader_id="quality",
    graders=[
        (JsonSchemaGrader("shape", schema=SCHEMA), 1.0),  # GATE by default
        (ActionGrader("action"), 1.0),
    ],
)
```

→ The full built-in catalog: [Grader Library](grader-library.md). The
gate-plus-judge pattern worked end to end:
[Evaluating a Real Agent §4](real-agent.md). Keeping an LLM judge honest:
[Human-Eval Calibration](human-eval.md).

---

## Decision 4 — Run it and read the results

`EvaluationRunner` drives the trials; `RunnerConfig` sets how many and how fast:

```python
from tracelens import EvaluationRunner, RunnerConfig

config = RunnerConfig(num_runs=5, max_concurrency=10, timeout_seconds=30.0)
batch = await EvaluationRunner(adapter, [composite], config).run(eval_set)
```

`run` is async — call it from `asyncio.run(...)`. For long suites, `RunnerConfig`
also takes a progress callback and a `checkpoint_path` so a rerun resumes
(`--progress` / `--checkpoint` on the CLI). `fail_fast=True` stops scheduling
new trials after the first `FAILED` or `INFRA_ERROR` trial — the rest are
recorded as `SKIPPED` — useful for smoke runs where one execution failure
means the harness is broken.

**Reading the `TrialBatch`.** Three things matter, in order:

1. **Harness vs. agent.** Check `batch.infra_error_rate` and
   `batch.grader_error_rate` *first*. A spike there means the eval broke, not the
   agent — don't trust the pass rate until those are near zero.
2. **Capability vs. reliability.** `batch.pass_rate` is the headline, but split it:
   `pass@k` (can it succeed at all in k tries?) and `pass^k` (does it succeed every
   time?). A high pass@k with a low pass^k is "capable but flaky."
3. **Is a change real?** To compare two runs, don't eyeball the means — use a
   bootstrap comparison.

Which statistic answers which question:

| Question | Use | Page |
|----------|-----|------|
| Can it do this at all? | `pass_at_k` | [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) |
| Is it reliable enough to ship? | `pass_to_k` | [pass@k vs pass^k](pass-at-k-vs-pass-hat-k.md) |
| Is version B actually better than A? | `compare_metrics` | [Comparing Versions](comparing-versions.md) |
| How confident are we in any number? | bootstrap CI | [Statistical Comparison](statistical-comparison.md) |

**Reports.** Hand the batch to `ReportGenerator` for markdown, JSON, HTML, or a
CI summary:

```python
from tracelens import ReportGenerator

gen = ReportGenerator(k_values=[1, 3, 5], consistency_k_values=[2, 3, 5])
report = gen.build_report(batch)
print(gen.render_ci_summary(report))   # also render_markdown / render_html
```

**Gating CI on regressions** — once a run looks good, freeze it as a baseline and
block future runs that decline:
[Baseline Regression Tutorial](baseline-regression-tutorial.md) and
[CI/CD Integration](ci-cd-integration.md).

---

## From the CLI

The same four decisions map to flags. The CLI loads your adapter and graders by
dotted import path (so they must be importable and constructible with no
arguments):

```bash
tracelens run \
  --eval-set eval/tasks.json \
  --adapter myproject.eval.MyAdapter \
  --graders myproject.eval.MyGrader \
  --num-runs 5 \
  --report reports/results.md \
  --html-report reports/results.html \
  --save-trials reports/trials.json
```

Add `--baseline-check --baselines-file eval/baselines.json --fail-on-regression
moderate` to gate CI, and `--progress` / `--checkpoint path.json` for long runs.
`tracelens report --results results.json --format markdown` re-renders a saved
run.

---

## Where to go next

- [Core Concepts & Glossary](concepts.md) — the object model these decisions act on.
- [Evaluating a Real Agent](real-agent.md) — all four decisions, worked end to end.
- [Grader Library](grader-library.md) · [Comparing Versions](comparing-versions.md) ·
  [Reproducibility & DecisionSpec](reproducibility.md) — the deep dives.
- [API Reference](reference.md) — every public class and function.
