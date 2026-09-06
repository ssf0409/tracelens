# CI/CD Integration Guide

This guide shows how to run TraceLens from CI with regression blocking,
artifact uploads, and human-eval calibration. TraceLens stays generic: your
project supplies the adapter, graders, task data, baselines, and rollout policy.

## Prerequisites

Add TraceLens to the project being evaluated:

```toml
[project]
dependencies = [
    "tracelens>=0.3.0",
]
```

If your project uses `uv`, CI installs it from the lockfile so the evaluated
dependency set is the one you tested locally:

```yaml
- uses: actions/checkout@v6
- uses: astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990 # v8.3.2
- name: Set up Python
  run: uv python install 3.12
- name: Install dependencies
  run: uv sync --frozen
```

`tracelens init` generates a `tracelens.yaml` and a workflow that runs
`tracelens run --config tracelens.yaml`. For projects that do not (yet) list
TraceLens as a dependency, the workflow creates an environment and installs
TraceLens pinned to the release that generated it. See `eval/README.md` in the
generated scaffold.

## Project Contract

The TraceLens CLI loads classes by dotted import path and instantiates them
with no constructor arguments:

```bash
tracelens run \
  --eval-set eval/tasks.json \
  --adapter myproject.eval.adapters.CIAgentAdapter \
  --graders myproject.eval.graders.CIQualityGrader
```

That means CI-facing adapters and graders should read configuration from the
task input, checked-in config files, or environment variables.

The same run can live in a committed `tracelens.yaml` and be invoked as
`tracelens run --config tracelens.yaml`: flags on the command line override
the file, paths in the file resolve relative to it, and adapters import from
the file's directory unless `run.import_root` says otherwise. The
[run configuration file](user-guide.md#run-configuration-file) reference
lists every key.

Minimal adapter shape:

```python
from tracelens import AgentAdapter, Task, Transcript


class CIAgentAdapter(AgentAdapter):
    async def run(self, task: Task) -> Transcript:
        transcript = self.start_transcript(task)
        transcript.final_output = await call_agent(task.input_data)
        return transcript
```

Minimal grader shape:

```python
from tracelens import CodeGrader, Task, Transcript


class CIQualityGrader(CodeGrader):
    def __init__(self) -> None:
        super().__init__(grader_id="ci_quality")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        expected = task.metadata["expected"]
        actual = transcript.final_output.get("answer")
        return {"correct": 1.0 if actual == expected else 0.0}

    def determine_pass(
        self, metrics: dict[str, float], task: Task
    ) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]
```

## Pull Request Workflow

A ready-to-copy version of the workflow below lives at
[`examples/ci/eval.yml`](https://github.com/ssf0409/tracelens/blob/main/examples/ci/eval.yml) — drop it at
`.github/workflows/eval.yml` and edit the five marked placeholders.

Create `.github/workflows/eval.yml` in the downstream project:

```yaml
name: TraceLens Evaluation

on:
  pull_request:
    branches: [main]
    # Every pull request is evaluated by default so agent code changes never
    # skip the eval. To narrow it, list the paths whose changes should
    # trigger an eval -- and include your agent's source directories.
    #   paths:
    #     - "app/**"
    #     - "eval/**"
    #     - "pyproject.toml"
    #     - "uv.lock"
  workflow_dispatch:

permissions:
  contents: read

jobs:
  eval:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v6

      - uses: astral-sh/setup-uv@11f9893b081a58869d3b5fccaea48c9e9e46f990 # v8.3.2

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --frozen

      - name: Run TraceLens
        run: |
          uv run --frozen tracelens run \
            --eval-set eval/tasks.json \
            --adapter myproject.eval.adapters.CIAgentAdapter \
            --graders myproject.eval.graders.CIQualityGrader \
            --num-runs 5 \
            --baseline-check \
            --baselines-file eval/baselines.json \
            --fail-on-regression moderate \
            --output eval/results/results.json \
            --report eval/results/report.md \
            --html-report eval/results/report.html \
            --save-trials eval/results/trials.json

      # The report exists only after a successful preflight; a missing file
      # must not turn a clear TraceLens error into a `cat` failure.
      - name: Add report to job summary
        if: always()
        run: |
          if [ -f eval/results/report.md ]; then
            cat eval/results/report.md >> "$GITHUB_STEP_SUMMARY"
          else
            echo "No TraceLens report was written; see the run step for the error." \
              >> "$GITHUB_STEP_SUMMARY"
          fi

      - name: Upload evaluation artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: tracelens-results
          if-no-files-found: ignore
          path: |
            eval/results/results.json
            eval/results/report.md
            eval/results/report.html
            eval/results/trials.json
```

Three things this workflow gets right that are easy to get wrong:

1. **Triggers cover agent code.** It runs on every pull request. A `paths:`
   filter is a project decision; if you add one, list your agent's source
   directories, or a change that only touches the agent will skip the eval
   and merge with a green check that never ran.
2. **Installs are reproducible.** `uv sync --frozen` and `uv run --frozen`
   install exactly what `uv.lock` records, so CI evaluates the dependency
   set you tested. Commit the lockfile.
3. **Failures stay visible.** The run step's exit code is the job's result.
   The summary and artifact steps run `if: always()` but tolerate files that
   a failed preflight never wrote, so the original TraceLens error is what
   you read, not a `cat` failure.

## Reading the gate

`tracelens run --baseline-check` makes one decision per run and records it in
every output: the exit code, the summary on stdout, and a `gate` object in the
`--output` JSON, which the Markdown and HTML reports render and
`tracelens report` re-renders without recomputing anything.

| Status | Exit code | Meaning | What to do |
|--------|-----------|---------|------------|
| `not_requested` | 0 | The run had no `--baseline-check`. | Nothing; no gate was evaluated. |
| `passed` | 0 | At least one task was compared and nothing blocked. | Merge. |
| `blocked` | 1 | A regression at or above `--fail-on-regression`, or `--require-baselines` with a task that has no baseline. | Read the regression table, decide whether the change is acceptable, then fix the agent or promote the baseline. |
| `unevaluable` | 2 | No task could be compared, or a baseline-backed task had no gradable trials, no comparable metric, or content that changed since its baseline was stored. Missing evidence never passes. | Fix the harness failure or baseline mismatch named in the reasons (re-store the baseline of an edited task), then rerun. |

A misconfigured gate (missing `--baselines-file`, an unreadable baselines
file, or a gate-only flag without `--baseline-check`) also exits 2, before
the eval runs. The codes follow the CLI-wide contract (0 success, 1 negative
result, 2 usage/input error or unevaluable) described in the
[User Guide](user-guide.md#exit-codes-and-error-output); pass `--debug` for
tracebacks on input errors. `unevaluable` takes precedence over `blocked`: an observed
regression is still recorded, but missing evidence cannot authorize either
verdict.

The `gate` object in `results.json` (abridged):

```json
"gate": {
  "status": "blocked",
  "exit_code": 1,
  "threshold": "moderate",
  "noise_band": 0.03,
  "require_baselines": false,
  "checked": 2,
  "skipped_no_baseline": 0,
  "skipped_no_gradable": 0,
  "skipped_no_comparable_metrics": 0,
  "skipped_task_content_changed": 0,
  "blocking_regressions": 1,
  "reasons": ["1 blocking regression(s) at threshold 'moderate': t-fail (severe)"],
  "tasks": [
    {
      "task_id": "t-fail",
      "outcome": "checked",
      "compared_trials": 2,
      "excluded_trials": 0,
      "blocking": true,
      "regressions": [
        {"metric_name": "pass_rate", "baseline_mean": 0.9, "current_mean": 0.0,
         "delta_percent": -100.0, "severity": "severe", "is_significant": true,
         "within_noise_band": false}
      ]
    }
  ]
}
```

Each task's `outcome` is one of `checked`, `no_baseline`,
`no_gradable_trials`, `no_comparable_metrics`, or `task_content_changed`,
with a `reason` when it was not checked. The last one means the task's
content hash (recorded in the run's [provenance](reproducibility.md#run-provenance)
and as `task_summaries[].task_hash`) no longer matches the `task_hash` stored
on its baseline: a task edited after baselining is never compared by id
alone. Re-store that baseline to clear it; baselines without a `task_hash`
are compared as before, with a warning naming them. To look at a saved
decision again:

```bash
tracelens report --results eval/results/results.json --format markdown
```

A results file written before gate decisions were recorded has no `gate`
key; `tracelens report` renders it without inventing one.

### Details

- `--baseline-check` requires `--baselines-file`, and the file must exist —
  a missing flag or file exits 2 before the eval runs, so a misconfigured
  gate can never pass vacuously. Exit 0 means an evaluable gate passed;
  exit 1 means it blocked on a regression or missing required baselines;
  exit 2 means the gate was misconfigured or could not be evaluated.
- Tasks with no stored baseline are skipped with a stderr warning and
  counted in the gate summary line (`N checked, M skipped (no baseline),
  K blocking regression(s)`). Add `--require-baselines` to fail instead
  when any task lacks a baseline. At least one task must have a comparable
  baseline: an empty suite, zero runs, or no matching baselines exits 2.
- Trials that failed for harness reasons (`INFRA_ERROR` status or a
  grader crash) are excluded from the baseline comparison — they surface
  via `infra_error_rate` / `grader_error_rate` and a per-task exclusion
  note instead of dragging pass-rate samples to zero. A task with no
  gradable trials left is counted as `skipped (no gradable trials)` and
  makes the gate `UNEVALUABLE` (exit 2), even if another task passed or
  regressed. Fix the harness failure and rerun before relying on the gate.
  Partial trial loss remains allowed when each baseline-backed task retains
  at least one gradable sample; this does not imply adequate statistical
  power. Agent failures and runner timeouts remain comparison observations.
- The current CLI compares task-level `pass_rate` and `mean_score` against
  baselines. Each baseline-backed task must share at least one of these
  metrics, otherwise it is counted as `skipped (no comparable metrics)` and
  the gate exits 2. An unevaluable check takes precedence over exit 1, while
  still printing any observed regressions and exclusion counts. Diagnostic
  output files requested with `--output` / `--save-trials` are still written.
- Pass `--decision-spec run_spec.json` (or stamp `DecisionSpec` on
  transcripts in your adapter) and store each baseline with its
  `decision_spec` to enable infra-noise-aware comparison: sub-noise-band
  regressions under a mismatched infra config are flagged but not
  blocking. Tune the band with `--noise-band` (default 0.03).
- `--infra-exceptions builtins.OSError myproject.errors.RateLimitError`
  extends which exception types count as `INFRA_ERROR` instead of agent
  failures — downstream policy, conservative by default.
- Use GitHub job summaries or artifacts for reports. `tracelens report`
  supports `markdown`, `json`, and `html` output formats.
- On flaky CI infrastructure, add `--max-infra-retries 2`: trials that end in
  `INFRA_ERROR` (network drops, OOM kills) are re-attempted with exponential
  backoff before counting against `infra_error_rate`. Agent failures and
  timeouts never retry, so this cannot inflate the pass rate.
- For long suites, `--checkpoint path.json` persists progress; re-running the
  job with the same file resumes, skipping completed trials and re-running
  infra-errored ones. Checkpoints are bound to the eval set, adapter, and
  graders that produced them — resuming with a mismatched or corrupt file
  fails with a clear error rather than mixing results from different runs.

## Baseline Files

Baselines are JSON files managed by `BaselineManager`. Create or update them in
a small project script, then commit the resulting JSON:

```python
from tracelens import BaselineManager, BaselineType, PromotionPolicy, TaskBaseline

manager = BaselineManager("eval/baselines.json")

baseline = TaskBaseline(
    task_id="math-add",
    task_name="Simple addition",
    baseline_type=BaselineType.CAPABILITY,
    promotion_policy=PromotionPolicy(
        allow_auto_promotion=True,
        min_improvement_relative=0.05,
        min_samples=10,
        required_confidence=0.95,
    ),
)
baseline.add_metric(
    metric_name="pass_rate",
    value=0.92,
    std=0.03,
    sample_size=30,
    relative_threshold=0.05,
)
baseline.add_metric(
    metric_name="mean_score",
    value=0.88,
    std=0.04,
    sample_size=30,
    relative_threshold=0.05,
)

manager.set_baseline(baseline)
manager.save()
```

When the values come from a results file, also store the task's content hash
(`TaskBaseline(..., task_hash=summary["task_hash"])` from
`task_summaries[]`, or `update_baseline(..., task_hash=...)`). The gate then
compares that task only while its content still matches and refuses, rather
than silently matching on id, once the task is edited; see
[Run provenance](reproducibility.md#run-provenance).

Use canary baselines for protected floors that must not auto-promote:

```python
from tracelens import BaselineManager

manager = BaselineManager("eval/baselines.json")
manager.create_canary_baseline(
    task_id="safety-critical-task",
    metrics={"pass_rate": 1.0, "mean_score": 1.0},
    fingerprint="decision-spec-fingerprint",
    sample_size=20,
)
manager.save()
```

TraceLens does not auto-commit baseline updates. Keep that policy in your
downstream project so reviewers can decide when a changed baseline is
intentional.

## Scheduled Calibration

LLM-as-judge graders should be calibrated against human judgement. The current
TraceLens loop is:

```bash
tracelens run \
  --eval-set eval/tasks.json \
  --adapter myproject.eval.adapters.CIAgentAdapter \
  --graders myproject.eval.graders.CIQualityGrader \
  --num-runs 5 \
  --save-trials eval/results/trials.json

tracelens sample \
  --trials eval/results/trials.json \
  --size 20 \
  --strategy diverse \
  --output eval/human-review/review.json
```

The generated `review.json` is a fill-in worksheet. TraceLens deliberately does
not ship a rating UI or human-grade store; use a spreadsheet, form, notebook, or
internal review tool to fill `human_score` and `human_passed`.

After a reviewer fills the worksheet:

```bash
tracelens reconcile \
  --annotations eval/human-review/review.json \
  --threshold 0.7 \
  --output eval/human-review/calibration.json
```

`reconcile` exits non-zero when Pearson correlation is below the threshold, so
you can run it as a scheduled alert or a release gate.

## Optional Weekly Workflow

```yaml
name: TraceLens Calibration

on:
  schedule:
    - cron: "0 0 * * 0"
  workflow_dispatch:

jobs:
  sample:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv python install 3.12
      - run: uv sync

      - name: Run evaluation and keep trials
        run: |
          uv run tracelens run \
            --eval-set eval/tasks.json \
            --adapter myproject.eval.adapters.CIAgentAdapter \
            --graders myproject.eval.graders.CIQualityGrader \
            --num-runs 5 \
            --save-trials eval/results/trials.json

      - name: Select human-review sample
        run: |
          uv run tracelens sample \
            --trials eval/results/trials.json \
            --size 20 \
            --strategy diverse \
            --output eval/human-review/review.json

      - name: Upload worksheet
        uses: actions/upload-artifact@v4
        with:
          name: human-review-worksheet
          path: eval/human-review/review.json
```

## Handling Intentional Regressions

If a regression is intentional, prefer an explicit review decision over hiding
it in CI. Common patterns:

- Add a PR note explaining which metric regressed and why.
- Commit a reviewed baseline update in the same PR.
- Keep canary baselines manual-only for safety-critical tasks.

Avoid broad "accept all regressions" switches. The value of TraceLens in CI is
that it makes lower-quality behavior visible before merge.
