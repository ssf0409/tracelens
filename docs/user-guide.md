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
(`--progress` / `--checkpoint` on the CLI). Resume skips completed trials but
re-runs infra-errored ones, and refuses (with `CheckpointError`) a checkpoint
written by a different eval set, adapter, graders, or `DecisionSpec` —
identity is class-path based, so pass a `DecisionSpec` to distinguish two
configs of the same adapter class, and use stable explicit `task_id`s
(auto-generated ids change every run and can never resume). On flaky
infrastructure,
`max_infra_retries` re-attempts `INFRA_ERROR` trials with exponential backoff —
agent failures and timeouts never retry, so retries can't inflate the pass rate.
`fail_fast=True` stops scheduling new work after the first trial whose
execution fails (`FAILED`, `INFRA_ERROR` after retries are exhausted, or
`TIMEOUT`) — useful for smoke runs where one execution failure means the
harness is broken. In-flight trials finish normally, unstarted work simply
never runs (no placeholder trials, so pass rates and the baseline gate only
see trials that actually executed), and a grading failure or a teardown
error never trips it.
**Reading the `TrialBatch`.** Three things matter, in order:

1. **Harness vs. agent.** Check `batch.infra_error_rate` and
   `batch.grader_error_rate` *first*. A spike there means the eval broke, not the
   agent — don't trust the pass rate until those are near zero.
2. **Capability vs. reliability.** `batch.pass_rate` is the headline, but split it:
   `pass@k` (can it succeed at all in k tries?) and `pass^k` (does it succeed every
   time?). A high pass@k with a low pass^k is "capable but flaky."
3. **Is a change real?** To compare two runs, don't eyeball the means — use a
   bootstrap comparison.
4. **Why did a trial fail?** Read the trial, not the number:
   `tracelens inspect trials.json --failures` shows the kind of failure,
   expected versus actual, grader feedback, and the transcript
   ([Debugging a Failed Evaluation](inspecting-failures.md)).

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

The same four decisions map to flags, or to a committed `tracelens.yaml` (see
[Run configuration file](#run-configuration-file) below). The CLI loads your
adapter and graders by dotted import path (so they must be importable and
constructible with no arguments):

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
moderate` to gate CI. The gate exits 1 when it blocks on a regression or missing
required baselines, and 2 when it is misconfigured or unevaluable. Missing
baseline files fail before execution; zero checked tasks or any baseline-backed
task with no gradable trials or comparable metrics, or whose content changed
since its baseline was stored, makes the check unevaluable after execution.
Exit 2 takes precedence over observed policy failures. The
gate always prints a summary of what it checked. Tasks with no stored baseline
are skipped with a warning when other tasks can be checked; add
`--require-baselines` to fail instead. Use `--progress` / `--checkpoint
path.json` / `--max-infra-retries N` for long runs. See
[CI/CD Integration](ci-cd-integration.md) for the noise-aware flags
(`--decision-spec`, `--noise-band`, `--infra-exceptions`).
`tracelens report --results results.json --format markdown` re-renders a saved
run.

`tracelens compare baseline-trials.json candidate-trials.json` decides whether
a second run of the same eval set is better, worse, or indistinguishable, from
the two `--save-trials` files. It pairs each task's statistic across the runs,
bootstraps over tasks, and reports a verdict against `--threshold` (default
0.03): exit 0 for improvement, equivalence, or a significant-but-negligible
change; 1 for a regression; 2 when the runs are not comparable (changed task
content, different graders) or the evidence is insufficient or inconclusive.
`--metric pass_rate|mean_score|<grader_id>.<metric_name>`, `--direction lower`
for metrics like latency, `--unmatched-tasks exclude` to compare only shared
tasks, `--observe` to always exit 0, and `--output compare.json` for the
record. See [Comparing Versions](comparing-versions.md) and the
[statistical contract](statistical-contract.md#run-versus-run-comparison-tracelens-compare-issue-28).

`tracelens inspect eval/results/trials.json --failures` explains a failed run
from its trials file: each failing trial's kind (agent failure, infra error,
or grader crash, never conflated), expected versus actual (with
`--eval-set`), grader feedback, and transcript steps, bounded with explicit
omission counts (`--full` lifts the bounds; `--html` writes an offline
drilldown). Fix and rerun only the affected tasks with
`tracelens run ... --task-id ID` (`run.task_ids` in a config file). See
[Debugging a Failed Evaluation](inspecting-failures.md).

### Run configuration file

Commit the run settings instead of repeating flags in every README and CI
step. `tracelens init` writes a `tracelens.yaml`, and
`tracelens run --config tracelens.yaml` runs it from any directory. Every key
is a `run` flag; this file lists all of them:

```yaml
run:
  eval_set: eval/tasks.json          # --eval-set (.json, .jsonl, .csv, or a directory)
  eval_set_format: json              # --eval-set-format (required for a directory)
  input_field: input                 # --input-field (jsonl/csv column with the input)
  metadata_fields: [difficulty]      # --metadata-fields (jsonl/csv columns to keep)
  adapter: eval.adapter.MyAdapter    # --adapter
  graders: [eval.grader.MyGrader]    # --graders
  task_ids: [math-add]               # --task-id (targeted rerun; omit to run every task)
  import_root: .                     # dotted paths import from here (default: this directory)
  num_runs: 5                        # --num-runs
  max_concurrency: 5                 # --max-concurrency
  timeout: 300                       # --timeout, in seconds
  progress: true                     # --progress / --no-progress
  checkpoint: eval/results/checkpoint.json   # --checkpoint
  max_infra_retries: 0               # --max-infra-retries
  infra_exceptions: [builtins.OSError]       # --infra-exceptions
  decision_spec: eval/decision-spec.json     # --decision-spec
  outputs:
    results: eval/results/results.json       # --output
    report: eval/results/report.md           # --report
    html_report: eval/results/report.html    # --html-report
    trials: eval/results/trials.json         # --save-trials
  baseline:
    enabled: true                    # --baseline-check / --no-baseline-check
    file: eval/baselines.json        # --baselines-file
    fail_on_regression: moderate     # --fail-on-regression
    require_baselines: false         # --require-baselines / --no-require-baselines
    noise_band: 0.03                 # --noise-band
```

Every key is optional, but some layer must provide `eval_set`, `adapter`,
and `graders`. The rules are fixed:

- **Precedence.** Built-in defaults, then the file, then the flags you type;
  each layer overrides the one before. An omitted flag never resets a value
  from the file, and booleans override in both directions (`--no-progress`
  beats `progress: true`; `--no-baseline-check` switches a configured gate
  off for one run).
- **Paths.** Paths in the file resolve relative to the file. Paths given as
  flags resolve against the current directory, as they always have. The
  `[tracelens] wrote ...` lines on stderr show the resolved locations.
- **Imports.** Adapters and graders are imported from `run.import_root`, by
  default the file's directory, so the command behaves the same from any
  working directory. TraceLens never changes the process directory.
- **Strictness.** The file is read with YAML's safe loader and validated
  before any agent call: an unknown key, a duplicate key, a wrong type, a
  bad enumeration, or an unsafe YAML construct exits 2 with a message that
  names the file and the dotted key. There are no profiles, includes,
  matrices, or variable interpolation, and secrets belong in environment
  variables, not in the file.

---

### Exit codes and error output

Every command follows one contract, so a CI step can branch on the code
without parsing text:

| Exit code | Meaning | Examples |
|-----------|---------|----------|
| `0` | Success, or the gate passed | a run completed; `reconcile` found the grader calibrated; `compare` found no regression; `inspect` printed its report |
| `1` | A negative result | the baseline gate blocked; `--require-baselines` unmet; Pearson r below `--threshold`; `compare` found a regression |
| `2` | A usage, configuration, or input error, or a gate that could not be evaluated | missing or unreadable input file, invalid JSON, an unimportable adapter or grader, `--num-runs 0`, a gate with no gradable trials, `init` refusing to overwrite without `--force`, a `compare` of runs that are not comparable or whose evidence is inconclusive |

Input and configuration problems are reported before any agent call, as one
or two lines on stderr: the message names the file (and the line, where the
loader knows it) and the next action. Pass `--debug` (before the subcommand,
`tracelens --debug run ...`) or set `TRACELENS_DEBUG=1` to add the full
traceback. Unexpected programming failures are never swallowed.

Streams are separated so scripts can rely on them: stdout carries only the
result (the run summary and gate lines, a rendered report, the sampled
worksheet), while progress, warnings, and the list of written artifacts go to
stderr:

```text
[tracelens] wrote results: reports/results.json
[tracelens] wrote report: reports/results.md
[tracelens] wrote trials: reports/trials.json
```

## Where to go next

- [Core Concepts & Glossary](concepts.md) — the object model these decisions act on.
- [Evaluating a Real Agent](real-agent.md) — all four decisions, worked end to end.
- [Grader Library](grader-library.md) · [Comparing Versions](comparing-versions.md) ·
  [Reproducibility & DecisionSpec](reproducibility.md) — the deep dives.
- [API Reference](reference.md) — every public class and function.
