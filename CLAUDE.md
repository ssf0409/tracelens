# TraceLens - Development Guide

## Project Overview

TraceLens is an open source evaluation and regression-testing framework for AI
agents. It turns agent runs into inspectable traces, graded outcomes, baseline
comparisons, calibration data, and CI-ready reliability signals.

Keep this repository domain-agnostic. Checked-in docs and examples should work
for external users without private project names, local absolute paths, or
unpublished downstream integrations.

## Ownership Boundary

### TraceLens Owns

- Core data models: `Task`, `EvalSet`, `Trial`, `Transcript`, `Outcome`
- Execution primitives: `AgentAdapter`, `SimpleAdapter`, `HTTPAPIAdapter`,
  `EvaluationRunner` (concurrency, timeouts, progress, checkpoint/resume)
- Grader abstractions: `CodeGrader`, `LLMGrader`, `CompositeGrader`
- Built-in validators and budget/event-chain graders
- Statistical analysis: `pass@k`, `pass^k`, bootstrap confidence intervals
- Baseline management and regression detection
- Report rendering for markdown, JSON, HTML, and CI summaries
- Human-eval calibration: sample trial worksheets and reconcile human vs grader
  scores
- Reproducibility fingerprints via `DecisionSpec`

### Downstream Projects Own

- Domain task data and eval-set curation
- Agent invocation details and adapter subclasses
- Domain-specific graders and thresholds
- Baseline files and promotion policy
- CI policy for blocking, warning, or manual review

TraceLens evaluates evidence; it should not become the source of domain truth
for a downstream project.

## Key Files

```
src/tracelens/
├── core/
│   ├── task.py          # Task, TaskLoader, EvalSet - test case definitions
│   ├── trial.py         # Trial, TrialBatch - execution tracking
│   ├── grader.py        # Grader ABCs - CodeGrader, LLMGrader, CompositeGrader
│   ├── transcript.py    # Transcript - execution record
│   ├── decision_spec.py # DecisionSpec - reproducibility fingerprinting
│   ├── provenance.py    # RunProvenance - run identity, compatibility check
│   └── outcome.py       # Outcome - grading result (incl. grader_error flag)
├── execution/
│   ├── runner.py        # EvaluationRunner - parallel execution, checkpoint/resume
│   ├── agent_adapter.py # AgentAdapter ABC, SimpleAdapter
│   ├── http_adapter.py  # HTTPAPIAdapter for JSON endpoints
│   └── registry.py      # Plugin loading via dotted import paths
├── statistics/
│   ├── pass_at_k.py     # pass@k - capability ceiling
│   ├── consistency.py   # pass^k - reliability measurement
│   ├── inference.py     # Bootstrap CI, significance testing
│   ├── latency.py       # Latency aggregation helpers
│   └── run_comparison.py # compare_runs - paired task bootstrap (tracelens compare)
├── baselines/
│   ├── manager.py       # BaselineManager - store/retrieve/promote baselines
│   └── comparison.py    # RegressionDetector - detect regressions
├── calibration/
│   ├── analyzer.py      # CalibrationAnalyzer - grader vs human agreement
│   └── sampler.py       # sample_for_review - select trials for human review
├── contracts/
│   └── contract.py      # BehaviorContract - declarative grader generation
├── graders/
│   └── event_chain.py   # Event-chain verifier
├── llm/
│   ├── provider.py      # LLMProvider ABC and InMemoryProvider
│   └── factory.py       # Provider factory policy
├── metrics/
│   ├── budgets.py       # Latency/token/tool-call/trace consistency graders
│   └── validators.py    # JSON schema, regex, contains, constraint graders
├── reporting/
│   ├── generator.py     # ReportGenerator - markdown, JSON, HTML, CI summary
│   └── inspect.py       # Failure inspection views: text, HTML, JSON
└── cli/
    ├── main.py          # run / report / sample / calibrate / reconcile
    ├── config.py        # tracelens.yaml run configuration (run --config)
    ├── init.py          # Scaffold: eval/, tracelens.yaml, CI workflow
    ├── compare.py       # tracelens compare - two saved runs, paired task bootstrap
    ├── inspect.py       # tracelens inspect - explain failed trials from a trials file
    ├── sample.py        # Human review worksheet generation
    └── calibrate.py     # Human-vs-grader reconciliation
```

## Grader Types

### CodeGrader

Use deterministic, code-based grading when expected outputs or measurable
metrics are available. It should produce the same result for the same
transcript and task.

```python
class ExactAnswerGrader(CodeGrader):
    def compute_metrics(self, transcript, task) -> dict[str, float]:
        expected = task.metadata["expected"]
        actual = transcript.final_output.get("answer")
        return {"correct": 1.0 if actual == expected else 0.0}

    def determine_pass(self, metrics, task) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]
```

### LLMGrader

Use LLM-as-judge grading for subjective quality dimensions such as helpfulness,
specificity, reasoning quality, or rubric adherence. LLM graders are
non-deterministic, so they need calibration against human judgement.

`GraderConfig` controls per-attempt timeout (`timeout_seconds`) and retry on
transient or malformed responses (`retry_on_error`, `max_retries`,
`retry_backoff_seconds`).

```python
class HelpfulnessGrader(LLMGrader):
    def build_grading_prompt(self, transcript, task) -> str:
        return f"Evaluate helpfulness on a 0-1 scale:\n{transcript.final_output}"

    def parse_llm_response(self, response, task):
        return passed, score, metrics, feedback
```

### CompositeGrader and BehaviorContract

Use `CompositeGrader` when some graders are hard gates and others contribute to
score. Use `BehaviorContract.to_graders()` when output rules can be declared as
a reusable contract.

A sub-grader crash is recorded as a `grader_error` outcome and surfaced
separately from agent failures (`TrialBatch.grader_error_count/rate`, also in
reports) — a spike there means the grading harness broke, not the agent.

## Statistical Analysis

Estimator definitions, sampling units, and trial-validity rules live in
`docs/statistical-contract.md`. Any change to them updates that page in the
same PR.

### pass@k (Capability)

"What's the probability of at least one success in k attempts?"

Use for capability evaluation: can the agent do this at all?

```python
pass_at_k(n=10, c=7, k=5)
```

### pass^k (Reliability)

"What's the probability that all k attempts succeed?"

Use for reliability evaluation: is the agent consistent enough to trust?

```python
pass_to_k(results=[True, True, False, True, True], k=3)
```

## Baseline Regression Detection

### Severity Levels

- **NONE**: No regression
- **MINOR**: Small decline
- **MODERATE**: Blocks CI by default in most projects
- **SEVERE**: Large decline that should block unless explicitly accepted

### Baseline Types

- **CANARY**: Protected floor; manual promotion only
- **CAPABILITY**: Tracks current capability; can auto-promote on improvement
- **EXPERIMENTAL**: Loose baseline for active exploration

### Reproducibility

Pass a `DecisionSpec` to `EvaluationRunner` and it is stamped onto every
transcript that doesn't already carry one, so baselines record a fingerprint
of the exact configuration that produced them.

Every run also records a `RunProvenance` (`batch.provenance`; `provenance` in
results and trials JSON; a "Run Provenance" report section): a `measurement`
side (per-task SHA-256 content hashes, grader identities with an optional
declared `provenance_version`, runner settings) and a `candidate` side
(adapter identity, `DecisionSpec` fingerprint). `check_compatibility(a, b)`
says whether two runs measured the same thing (compatible / incompatible /
unknown) and lists what changed on each side. The gate refuses to compare a
task whose content hash differs from the one stored on its baseline
(`TaskBaseline.task_hash`, taken from `task_summaries[].task_hash`) and
marks the run unevaluable. Hashing rule and semantics: `docs/reproducibility.md`,
"Run provenance".

## CLI Usage

```bash
tracelens run \
  --eval-set eval/suite.json \
  --adapter myproject.eval.adapters.MyAgentAdapter \
  --graders myproject.eval.graders.QualityGrader \
  --num-runs 5 \
  --output reports/results.json \
  --report reports/results.md \
  --html-report reports/results.html \
  --save-trials reports/trials.json

tracelens report --results reports/results.json --format markdown
```

`tracelens run --config tracelens.yaml` reads the same settings from a file
(`tracelens init` writes one). Precedence: built-in defaults, then the file,
then explicit flags; booleans have `--no-` forms. Paths in the file resolve
relative to the file, dotted imports come from `run.import_root` (default:
the file's directory), and the file is safe-loaded and validated strictly:
unknown or duplicate keys, wrong types, and missing required settings exit 2
before any agent call. Schema: `docs/user-guide.md`, "Run configuration
file".

`tracelens compare baseline-trials.json candidate-trials.json` implements the
contract's run-versus-run section (`tracelens.statistics.run_comparison`):
tasks aligned by content via provenance (incompatible runs exit 2;
`--unmatched-tasks exclude` compares shared tasks; legacy artifacts align by
id and are labelled), one statistic per task and run (`--metric pass_rate |
mean_score | <grader_id>.<metric_name>`, `--direction`, `--grader`), paired
differences with a task bootstrap and sign-flip p-value, and a verdict against
`--threshold`: improvement / equivalent / below-threshold exit 0, regression
1, inconclusive or insufficient evidence 2 (`--observe` forces 0). The stdout
summary and `--output` JSON share every field. Fixtures for the tests live in
`tests/fixtures/compare/` (regenerate with `generate.py`).

`tracelens inspect trials.json [--failures|--all|--kind ...] [--task-id ...]
[--grader ...] [--eval-set ...] [--html ...] [--json ...]` explains trials
from a `--save-trials` file (`tracelens.reporting.inspect`): one kind per
trial (agent failure / infra error / grader error / not run / passed, harness
causes first), expected versus actual (with `--eval-set`), grader verdicts and
feedback, transcript steps; every absent field reads `missing`, output is
bounded (400 chars per field, 20 steps) with omitted counts, `--full` lifts
the bounds. Exit 0 whenever the file was read; 2 for input errors. Targeted
reruns: `tracelens run --task-id ID ...` (`run.task_ids` in the config) runs
only those tasks; provenance and checkpoint identity then cover the subset.

Exit codes across all commands: 0 = success or gate passed; 1 = negative
result (blocked gate, unmet `--require-baselines`, calibration below
threshold); 2 = usage/configuration/input error (before any agent call) or an
unevaluable gate. `tracelens --debug ...` or `TRACELENS_DEBUG=1` adds
tracebacks to input errors; written artifact paths are listed on stderr, and
stdout carries only the result.

`--eval-set` accepts `.json`, `.jsonl`, and `.csv` files (format inferred from
the suffix; directories need `--eval-set-format`); `--input-field` and
`--metadata-fields` map foreign JSONL/CSV columns. Eval-set load failures
exit 2 before any agent call.

Baseline checks need a baseline file:

```bash
tracelens run \
  --eval-set eval/suite.json \
  --adapter myproject.eval.adapters.MyAgentAdapter \
  --graders myproject.eval.graders.QualityGrader \
  --baseline-check \
  --baselines-file eval/baselines.json \
  --fail-on-regression moderate
```

Gate semantics: a misconfigured check (missing `--baselines-file`, or the
file doesn't exist) exits 2 before the eval runs. After execution, exit 2
also marks an unevaluable gate: no tasks checked, or any baseline-backed task
with no gradable trials, no comparable CLI metrics, or task content that
changed since its baseline was stored. This takes precedence
over policy failures; otherwise exit 1 means a blocking regression or missing
required baseline. Exit 0 means an evaluable gate passed. Tasks without
baselines are warned and counted; `--require-baselines` makes them fail
instead, and zero matching baselines always makes the gate unevaluable. Noise-aware
comparison activates when both sides carry a `DecisionSpec` — via
`--decision-spec` or adapter-stamped transcripts, plus
`TaskBaseline.decision_spec`; tune the band with `--noise-band`.
`--infra-exceptions` extends which exception types count as `INFRA_ERROR`.
The decision is recorded once (`ReportData.gate`, a `GateResult` with status
`not_requested` / `passed` / `blocked` / `unevaluable`, per-task outcomes, and
the observed regressions), written to the `--output` JSON, rendered in every
report format, and re-rendered by `tracelens report`. Output-write failures
and non-results input to `report` exit 2 with a clear message.

Long runs: `--progress` prints per-trial progress to stderr, and
`--checkpoint path.json` persists trials periodically so a rerun with the same
path resumes — completed trials are skipped, infra-errored trials re-run.
Checkpoints record the eval-set content hash, adapter/grader identity, and
the run's `DecisionSpec` fingerprint (when one is set); resume requires
stable explicit `task_id`s;
resuming against a mismatched or corrupt checkpoint raises `CheckpointError`
instead of silently merging foreign trials. `--max-infra-retries N`
re-attempts `INFRA_ERROR` trials with exponential backoff (agent failures and
timeouts never retry); the attempt count is recorded on `Trial.attempts`.

## Human Evaluation Workflow

Periodic calibration catches LLM-grader drift:

1. `tracelens run ... --save-trials trials.json` - keep raw trials.
2. `tracelens sample --trials trials.json --size 20 --strategy diverse --output review.json` -
   select trials to hand-grade. Strategies: `diverse`, `boundary`, `failures`,
   `random`.
3. A human fills `human_score` / `human_passed` in `review.json`.
4. `tracelens reconcile --annotations review.json --threshold 0.7` - compare
   grader and human judgement. `reconcile` is an alias for `calibrate`.

TraceLens does not ship a rating UI or human-grade store.

## Testing

```bash
# Single-entry verification gate (lock check -> lint -> typecheck -> tests + coverage)
make verify
```

Individual steps (what `make verify` runs):

```bash
uv lock --check
uv run --frozen --extra dev ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous
uv run --frozen --extra dev mypy src/tracelens/
uv run --frozen --extra dev pytest -q --cov=tracelens --cov-fail-under=90
```

For packaging, CLI, README, public imports, or dependency metadata changes, also
run the built-wheel smoke path in `docs/contributor-testing.md`.

`tests/journey/test_user_journey.py` is the documented user journey run as
real `tracelens` processes (issue #33, Stage A): init, run --config,
baselines, gate, regression, inspect, compare, --task-id rerun, infra outage,
grader crash, malformed input, checkpoint/resume, report, sample. `make
verify` runs it against the editable install; CI's `user journey (built
wheel)` job runs it from a freshly built wheel via `TRACELENS_CLI`. Extend it
when a documented command changes.

## Documentation Rules

- Prefer public, runnable examples over private downstream project references.
- Use env vars for secrets and explicit CLI flags for behavior.
- Keep version constraints aligned with the latest published PyPI release.
- When showing CLI examples, verify them against `tracelens --help`.
- Distinguish illustrative data from live release/package data.

## Key Principles

1. **Grade outcomes, not paths** - Focus on what the agent produced.
2. **Handle non-determinism** - Use pass@k for capability and pass^k for
   reliability.
3. **Start with real failures** - Build suites from actual issues.
4. **Read transcripts** - Catch false signals and grader bugs.
5. **Calibrate regularly** - LLM graders drift without human calibration.
6. **Keep TraceLens general** - Domain truth and rollout policy belong in
   downstream projects.
7. **Separate harness failures from agent failures** - Track infra_error and
   grader_error rates alongside pass rates; a spike there means the eval is
   broken, not the agent.
