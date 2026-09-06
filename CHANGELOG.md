# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) —
but while `0.x`, minor bumps may contain breaking changes. Treat the
top-level `tracelens.*` imports as the stable surface; submodule paths may move.

## [Unreleased]

### Added

- **`tracelens inspect`: explain failed trials from a trials file.**
  `tracelens inspect eval/results/trials.json --failures` prints, per
  failing trial, one kind (agent failure, infra error, or grader crash,
  never conflated, harness causes first), status and attempts, expected
  versus actual output (`--eval-set` joins the task's name, input, and
  declared expectation), every grader's verdict, score, metrics, and
  feedback, and the transcript's steps, tokens, tool calls, and errors.
  Absent fields read `missing`; output is bounded (400 characters per
  field, 20 steps per transcript) with an explicit count of what was
  omitted, and `--full` lifts the bounds. Filters: `--kind agent|infra|
  grader|not-run|passed`, `--task-id`, `--grader` (trials that grader
  failed or crashed on), `--all`, `--limit`. `--html` writes a
  self-contained, escaped, offline drilldown that reads on a phone;
  `--json` writes the same view as data. The command exits 0 whenever the
  file was read (it reports, the gate decides) and 2 on input errors.
  `tracelens run --task-id ID ...` (`run.task_ids` in `tracelens.yaml`)
  reruns only the named tasks and refuses unknown ids; its provenance and
  checkpoint identity cover the subset. New guide: "Debugging a Failed
  Evaluation". (#52)
- **`tracelens compare`: a verdict between two saved runs.**
  `tracelens compare baseline-trials.json candidate-trials.json` (and
  `compare_runs()` in Python) implements the statistical contract's
  run-versus-run section: tasks are aligned by content through the runs'
  provenance (changed, added, or removed tasks and different graders make
  the runs incompatible; `--unmatched-tasks exclude` compares the shared
  tasks and lists the rest; artifacts without provenance align by id and
  are labelled, or refused with `--require-provenance`), one statistic per
  task and run is paired (`--metric pass_rate | mean_score |
  <grader_id>.<metric_name>`, `--direction lower` for latency-like metrics,
  `--grader` for multi-grader runs), and the mean paired difference gets a
  percentile bootstrap over tasks, a sign-flip p-value (exact for small
  suites), and a verdict against `--threshold`: improvement, equivalent
  within the threshold, or significant but below it exit 0; regression
  exits 1; inconclusive or insufficient evidence exits 2 (`--observe`
  forces 0). The terminal summary ("what changed" from the `DecisionSpec`
  diff next to "what moved" per task) and `--output compare.json` share
  every field, and the same inputs and `--seed` reproduce the record
  exactly. `examples/version_compare.py` now uses it. (#28)
- **Versioned run provenance and comparison compatibility.** Every
  `EvaluationRunner.run()` records a `RunProvenance` on the batch
  (`batch.provenance`; `provenance` in `--output` and `--save-trials` JSON;
  a "Run Provenance" section in Markdown and HTML): a `measurement` side
  (eval-set and per-task SHA-256 content hashes, grader identities with an
  optional declared `provenance_version`, runner settings) and a
  `candidate` side (adapter identity, `DecisionSpec` fingerprint and spec).
  `check_compatibility(a, b)` returns a `CompatibilityReport` that is
  `compatible`, `incompatible` (changed, added, or removed task content;
  different graders), or `unknown` (no provenance on a side), with runner
  and version differences as notes and candidate differences reported
  separately with a `DecisionSpec` diff. Baselines gain `task_hash`
  (`TaskBaseline`, `update_baseline(task_hash=)`, `promote(task_hash=)`;
  results carry `task_summaries[].task_hash`), and the CLI gate refuses to
  compare a task whose content changed since its baseline was stored
  (outcome `task_content_changed`, gate unevaluable, exit 2) instead of
  matching on id; baselines without a hash still compare, with a warning.
  Checkpoint identity now derives from the same hashing rule (values
  unchanged). Artifacts written before this release load with
  `provenance=None`; an unknown `schema_version` is rejected clearly. The
  `tracelens init` README snippet stores `task_hash` on each baseline. (#51)
- **`tracelens run --config tracelens.yaml`.** A project-owned run
  configuration file holds exactly what the `run` flags hold (eval set,
  adapter and graders, run counts, outputs, and the baseline gate).
  Precedence is built-in defaults, then the file, then flags given
  explicitly, so an omitted flag never resets a file value, and booleans
  override in both directions (`--progress` / `--no-progress`,
  `--baseline-check` / `--no-baseline-check`, `--require-baselines` /
  `--no-require-baselines`). Paths in the file resolve relative to the
  file; adapters and graders import from `run.import_root` (default: the
  file's directory) so the command works from any directory; and the file
  is parsed strictly with the safe YAML loader, so unknown keys, duplicate
  keys, wrong types, unsafe constructs, and missing required settings exit
  2 before any agent call. `tracelens init` now writes `tracelens.yaml`,
  and the generated README and workflow run the same
  `tracelens run --config tracelens.yaml`, so enabling the regression gate
  is one edit to the config file. (#35)
- **Actionable CLI errors and discoverable outputs.** `tracelens --debug` (or
  `TRACELENS_DEBUG=1`) adds the full traceback to input and configuration
  errors, which are otherwise one or two lines on stderr with the next
  action; an unimportable adapter or grader now explains the dotted-path
  and project-root requirement. `tracelens run` validates `--num-runs`,
  `--max-concurrency`, `--timeout`, and `--max-infra-retries` before doing
  anything, and lists every artifact it wrote on stderr
  (`[tracelens] wrote results: ...`) while stdout carries only the summary.
  (#48)
- **`tracelens run` accepts JSONL and CSV eval sets.** `--eval-set` picks
  the loader from the file suffix (`.json`, `.jsonl`, `.csv`); a directory
  needs `--eval-set-format json|jsonl|csv`. `--input-field` and
  `--metadata-fields` map foreign JSONL/CSV columns the same way the Python
  loaders do. The dispatch is also available as
  `tracelens.loaders.load_tasks()`, which raises `EvalSetLoadError` with the
  CLI's message. Hugging Face Hub datasets stay a Python-API concern. (#50)

### Changed

- **PyYAML is a core dependency** (`pyyaml>=6.0`), used only through the
  safe loader for `--config`. Flag-only `tracelens run` invocations are
  unchanged, except that `--eval-set`, `--adapter`, and `--graders` are now
  required only when no config file provides them; a run missing any of
  them still exits 2, naming both the flag and the config key. (#35)
- **One exit-code contract for every command.** 0 = success or gate passed;
  1 = a negative result (blocked gate, unmet `--require-baselines`,
  calibration below threshold); 2 = a usage, configuration, or input error,
  or an unevaluable gate. Consequently an unimportable adapter or grader,
  a missing / invalid / non-trials input to `tracelens sample`, a missing or
  invalid annotations, results, transcripts, or samples file for
  `tracelens calibrate` (and a worksheet with no usable rows or
  `--transcripts` without `--grader`/`--samples`), and `tracelens init`
  refusing to overwrite without `--force` all exit 2 instead of 1 (or a
  traceback). Negative results are unchanged. (#48)
- **Eval-set load failures exit 2.** A missing path, unsupported suffix,
  directory without `--eval-set-format`, invalid JSON, malformed record, or
  missing input column now prints a concise error naming the file (and the
  line when the loader knows it) and exits 2 before any agent call;
  previously these exited 1 or raised. (#50)
- **Report JSON records the baseline gate decision.** `tracelens run
  --output` now writes a `gate` object (status `not_requested` / `passed` /
  `blocked` / `unevaluable`, exit code, threshold, noise band, task counts,
  reasons, and per-task outcomes with the observed regressions), and
  `ReportData.gate` carries it in Python. Files written by earlier versions
  have no `gate` key and load with `gate = None`; no decision is invented.
- **`tracelens report` and output writing fail clearly.** `report` exits 2
  with a message for a missing results file, invalid JSON, or a document
  that is not a TraceLens results file (`ReportData.from_dict` now raises
  `ValueError` instead of rendering an empty report), and `tracelens run`
  exits 2 when an `--output` / `--report` / `--html-report` /
  `--save-trials` path cannot be written. Previously both produced
  tracebacks.
- **Gate comparisons use gradable trials only.** The per-trial samples fed
  to regression detection now follow `Trial.is_gradable`, so `PENDING`,
  `RUNNING`, and `SKIPPED` trials are excluded like harness failures
  (previously they counted as failures); `INFRA_ERROR` and grader-crash
  exclusion is unchanged.
- **Harness failures and never-run trials leave the pass-rate denominator.**
  `TrialBatch.pass_rate`, `passed_count`, `get_pass_results_by_task()`, and
  `get_pass_sequences_by_task()` now consider only *gradable* trials
  (`Trial.is_gradable`: `COMPLETED`, `FAILED`, and `TIMEOUT` without a grader
  crash). `INFRA_ERROR` trials, grader crashes, and `PENDING` / `RUNNING` /
  `SKIPPED` trials are excluded and appear as `None` gaps in run sequences;
  the new `gradable_count` / `excluded_count` properties and the report's
  `gradable_trials` field carry the denominator. Suite mean score likewise
  averages gradable trials only. The CLI baseline gate already excluded
  these trials, so gate decisions are unchanged; overall pass rates in
  reports rise for runs that had harness failures. Migration: batches that
  reported `pass_rate = passed / total_count` now report
  `passed / gradable_count`; use `batch.total_count - batch.gradable_count`
  (or `ReportData.excluded_trials`) to see what was excluded.
- **No more `c / n` fallback for pass@k below `k` runs.**
  `pass_at_k_estimator`, `PassAtKAnalyzer.analyze()`, and
  `compute_confidence_interval()` treat a task with fewer than `k` gradable
  runs as ineligible, matching `pass_to_k_estimator`. The float APIs return
  `0.0` when no task is eligible (their documented placeholder); use the new
  availability APIs below to tell that apart from a measured zero. Migration:
  code that relied on `pass_at_k_estimator({"t": [True]}, k=5) == 1.0`
  should either run `k` trials per task or read
  `pass_at_k_metric(...).value is None`.
- **Report JSON carries availability.** `ReportData.pass_at_k` and
  `reliability` values (and their per-task counterparts) may be `null` for
  an unavailable metric, and new top-level keys `metric_availability`,
  `availability_recorded`, and `gradable_trials` (plus `gradable_trials`
  per task summary) are written. Reports written by earlier versions still
  load: they get `availability_recorded = false`, their values are shown as
  recorded, and `gradable_trials` falls back to `total_trials`.

### Fixed

- **`tracelens init` generates a CI workflow that evaluates agent changes
  and installs reproducibly.** The generated `.github/workflows/eval.yml`
  used to trigger only on `eval/**`, `pyproject.toml`, and `uv.lock`, so a
  pull request that changed only agent code skipped the eval; it installed
  with `uv pip install tracelens` after `uv sync`, which `uv run` then
  removed again; it used older action refs than this repository's own CI;
  and its summary step failed on `cat` when a preflight error meant no
  report was written. It now runs on every pull request to `main` (a
  `paths:` filter is shown as an explicit customization), uses
  `actions/checkout@v6` and the same pinned `astral-sh/setup-uv` as this
  repository, installs an existing project from `uv.lock` with
  `uv sync --frozen` (or creates an environment in a bare repository) and
  installs TraceLens only when the project does not already provide it,
  pinned to the release that generated the file, runs
  `.venv/bin/tracelens` directly, tolerates missing report files in the
  summary and artifact steps, and carries the three gate flags as a
  commented block. The generated `eval/README.md` is a four-step
  walkthrough: run, what CI does, make it yours, and enable the gate with a
  baseline-storing snippet plus a "prove it blocks" step; an end-to-end test
  executes that walkthrough (init, run, store baselines from the README's
  own snippet, break the agent, confirm exit 1 and `BLOCKED`).
  `examples/ci/eval.yml` and the CI/CD guide are aligned with the template.
  (#49)
- **The gate decision is made once and shown everywhere.** `tracelens run
  --baseline-check` used to write JSON/Markdown/HTML before comparing against
  baselines, so a run that exited 1 on a severe regression saved artifacts
  that said nothing about it, and re-rendering with `tracelens report` lost
  regression data entirely. The comparison now runs first
  (`tracelens.reporting.gate.evaluate_gate`), the resulting `GateResult` is
  attached to the report before any file is written, and the exit code, the
  stdout summary, the JSON `gate` object, and the Markdown/HTML "Baseline
  Gate" sections (status, policy, task counts, reasons, a regression table
  with baseline/current/change/severity/notes, skipped tasks) all come from
  it. All-infra and all-grader-error runs stay distinguishable from agent
  regressions in every format, and grader-error rate and token totals are
  now rendered in Markdown, the CI summary, and HTML. (#47)
- **Unavailable metrics render as N/A, never as zeros.** A one-run-per-task
  suite used to report `pass@5 = 1.0` (a fallback) and `pass^5 = 0.0` (no
  eligible task) as if measured. New `MetricValue`
  (`tracelens.statistics`), `pass_at_k_metric`, `pass_to_k_metric`, and
  `PassAtKAnalyzer` / `ConsistencyAnalyzer.analyze_detailed()` return the
  value with its evidence: eligible/total task counts, the runs the metric
  needs, the most runs any task recorded, and a reason when unavailable.
  Markdown shows `N/A: needs at least 5 gradable runs per task; 0/2 tasks
  eligible; max 1 gradable run(s) recorded` plus a `--num-runs` hint, the
  CI summary prints `pass@5=n/a`, HTML lists unavailable metrics under the
  chart instead of drawing zero-height bars, and available values carry
  their eligible/total counts. Pass rates with no gradable trial render as
  `N/A` in every format, and per-task rows show `trials (gradable)` when
  they differ. Legacy reports get an explicit note that availability was
  not recorded. (#46)
- **pass^k no longer depends on trial completion order.** The runner appends
  trials as they finish, and `TrialBatch.get_pass_results_by_task()` returned
  them in that order, so the consecutive-window pass^k changed with
  concurrency timing and checkpoint resumes: run-index outcomes
  `[T, T, F, F]` reported pass^2 = 1/3 when trials finished in order and 0
  when they finished in order 0, 2, 3, 1. Results are now ordered by
  `run_index`. New `TrialBatch.get_pass_sequences_by_task()` returns
  run-indexed sequences with `None` for missing runs; `pass_to_k`,
  `pass_to_k_estimator`, and `ConsistencyAnalyzer` accept them, a window
  that would span a gap is not counted, and a task with no complete window
  is ineligible at that `k`. Reports use these sequences for pass^k. Two
  trials sharing a `(task_id, run_index)` now raise `ValueError` instead of
  producing an ambiguous sequence, and `pass_to_k` rejects `k < 1`. The
  `consistency.py` docstring examples that claimed
  `pass_to_k([T, T, F, T, T], 3) == 0.333` are corrected to `0.0`. pass@k
  is order-invariant and unaffected. (#45)
- **pass@k bootstrap intervals preserve repeated task draws.**
  `PassAtKAnalyzer.compute_confidence_interval` and `analyze_with_ci`
  resampled task IDs with replacement but collected them into a dict keyed
  by task ID, so a task drawn twice counted once: the draw `[A, A, B]` with
  A=1 and B=0 averaged to 1/2 instead of 2/3. Each resample lost about 37 %
  of its draws and reported intervals were roughly 20–25 % too narrow at
  typical suite sizes; intervals from earlier releases were overconfident
  and should be recomputed. Both methods now compute per-task pass@k once
  (in sorted `task_id` order) and delegate to `bootstrap_ci`, and gain an
  optional `seed` argument: the same inputs and seed give the same interval,
  and input task order no longer affects it. `bootstrap_ci`, and therefore
  the analyzer, now raises `ValueError` for `confidence` outside `(0, 1)` or
  `n_bootstrap < 1` instead of failing inside NumPy. Point estimates and the
  `(lower, upper)` return shape are unchanged. The estimator, sampling-unit,
  and trial-validity definitions every statistic follows are now written
  down in `docs/statistical-contract.md`. (#44)
- **Unevaluable baseline gates no longer pass.** `tracelens run
  --baseline-check` exits 2 when any baseline-backed task has no gradable
  trials or no comparable CLI metrics, or when no task can be checked at all
  (including empty suites, zero runs, and empty/unrelated baseline files).
  The summary marks the gate `UNEVALUABLE` and retains observed regressions
  and exclusion counts. This intentionally changes the previous exit-0
  behavior: infrastructure and grader failures remain excluded from agent
  regression samples, but missing evidence cannot authorize a passing gate.
  An unevaluable gate takes precedence over exit 1 for policy violations;
  otherwise regression and required-baseline failures still exit 1. Partial
  trial loss remains allowed when every baseline-backed task retains a
  gradable sample. Non-gated runs are unchanged.

## [0.4.0] - 2026-07-19

Reliability and data-portability release: CI gates now distinguish agent
regressions from harness noise, long runs retry and resume safely, and new
project scaffolding plus JSONL, CSV, and optional Hugging Face loaders shorten
the path from local data to a reproducible evaluation.

### Added

- **`tracelens init`.** New CLI command that scaffolds a runnable starter
  `eval/` suite, including tasks, adapter, grader, README, and a GitHub Actions
  workflow. The command refuses to overwrite generated files unless `--force`
  is provided.
- **Loud CI gate.** The baseline check now always prints a gate summary
  (`N checked, M skipped (no baseline), K blocking regression(s)`), warns
  per task when a baseline is missing, and `--require-baselines` turns
  missing baselines into a hard failure.
- **Configurable infra classification.** `RunnerConfig.infra_exception_types`
  (CLI: `--infra-exceptions`) extends which exception types are classified
  `INFRA_ERROR` instead of `FAILED`. The default set
  (`DEFAULT_INFRA_EXCEPTION_TYPES`) stays conservative: `InfraError`,
  `MemoryError`, `ConnectionError`.
- **Noise-aware gating from the CLI.** `TaskBaseline.decision_spec` stores
  the full spec alongside the fingerprint, `--decision-spec` loads the
  current run's spec (adapter-stamped transcripts work too), and the
  baseline check now runs `compare_with_specs()` — so sub-noise-band
  regressions under a mismatched infra config are flagged but not
  blocking, with the infra diff printed. `--noise-band` tunes the band.
- **DecisionSpec write path.** `update_baseline`,
  `create_capability_baseline`, `create_canary_baseline`, `promote`,
  `try_promote`, and `force_promote` all accept a `decision_spec`;
  creation derives the fingerprint from it when one isn't passed, and
  promotion refreshes the stored spec (archiving the old one in
  `previous_versions`) so it can't drift from the fingerprint.
- **`DEFAULT_INFRA_EXCEPTION_TYPES` is exported top-level** (`from
  tracelens import DEFAULT_INFRA_EXCEPTION_TYPES`), matching the
  documented `+ (OSError,)` extension pattern.
- **Infra-error retry.** `RunnerConfig.max_infra_retries` re-attempts trials
  that end `INFRA_ERROR`, with exponential backoff
  (`infra_retry_backoff_seconds`). `FAILED` and `TIMEOUT` trials never retry —
  those are observations about the agent, and retrying them would launder
  flakiness out of the pass rate. The final trial records its attempt count in
  `Trial.attempts`, and retried-away error messages are kept in
  `Trial.metadata["infra_retry_errors"]`. CLI: `--max-infra-retries`.
- **Checkpoint run identity.** Checkpoint files now carry a versioned envelope
  with the eval-set content hash, adapter/grader class identity, and the
  run-level `DecisionSpec` fingerprint when one is set (class paths alone
  cannot distinguish two configs of the same adapter class). Resuming
  against a checkpoint written by a different eval set, adapter, grader
  stack, or decision spec raises `CheckpointError` (exported from
  `tracelens`) instead of silently merging foreign trials keyed only on
  `(task_id, run_index)`. Envelopes with an unknown format version or a
  missing identity are rejected as corrupt. Note: resume requires stable
  explicit `task_id`s — auto-generated ids change every process.
  Pre-0.4 bare-batch checkpoints still load, with a loud warning that their
  identity can't be verified.
- **JSONL and CSV task loaders.** `JSONLTaskLoader` and `CSVTaskLoader`
  (top-level exports) load eval sets from `.jsonl`/`.csv` files or
  directories and save them back, with JSON-compatible round-trips (CSV
  serialises structured Task fields and one canonical metadata column as JSON) and
  no JSON coercion of free-text Task fields. Missing or ambiguous inputs,
  malformed CSV structure, and mixed canonical/flat metadata representations
  fail loudly. The optional `HFDatasetLoader` loads explicit Hub splits, supports
  revision pinning, and round-trips local saved datasets through the same mapping
  contract without adding a core dependency. Derived from #31 by @Balaji1304.
  Docs: `docs/task-sources.md`.

### Changed
- **Gate misconfiguration is now an error.** `tracelens run
  --baseline-check` without `--baselines-file`, or with a nonexistent or
  unparseable baselines file, exits 2 before the eval runs instead of
  silently skipping the entire regression check (the file is fully
  loaded during preflight, so a corrupt file can no longer burn a full
  eval before crashing). `--require-baselines` or `--noise-band` without
  `--baseline-check` is also an exit-2 usage error; `--baselines-file`
  alone warns that it has no effect.
- **Harness failures no longer masquerade as agent regressions in the
  gate.** The baseline check excludes `INFRA_ERROR` and grader-crash
  trials from the per-trial comparison samples (they remain visible via
  `infra_error_rate` / `grader_error_rate`, a per-task exclusion note,
  and a `skipped (no gradable trials)` count when nothing gradable
  remains). `TIMEOUT` trials still count against the agent.
- **Adapter-raised `TimeoutError` is no longer reported as a budget
  timeout.** Only the runner's own `asyncio.wait_for` budget produces
  `TrialStatus.TIMEOUT`; a `TimeoutError` from inside the adapter (e.g.
  `socket.timeout`) now classifies through `infra_exception_types`
  (`FAILED` by default, infra if configured) and keeps its original
  message.
- **Noise-downgraded reports are internally consistent.**
  `compare_with_specs()` now recomputes `overall_severity` from the
  blocking regressions and appends a noise-band note to the summary, so
  a noise-only report no longer reads `SEVERE` while
  `should_block_ci()` returns False. `should_block_ci(...,
  ignore_noise_band=False)` still counts every regression.
- **No more fabricated blocking on underpowered zero-variance samples.**
  A consistent drop that a valid z-test cannot call significant (e.g.
  five identical scores half a baseline standard deviation below the
  mean) no longer blocks CI — previously it always blocked via the
  fabricated p=0.0. Decisive drops still block; degenerate cases with no
  valid test still block on thresholds with `insufficient_data=True`.
- **No fabricated significance on degenerate samples.**
  `MetricRegression.p_value` is `None` (not `0.0`) when no valid test
  exists — n=1 with `baseline_std=0`, or zero variance on both sides. Such
  regressions are still reported and can still block CI, with severity
  from the delta thresholds and an explicit `insufficient_data` flag.
  Zero-variance samples against a known baseline spread now get a real
  z-test.
- **Checkpoint resume re-runs infra-errored trials.** Resume previously
  skipped every finished trial, permanently freezing `INFRA_ERROR` results
  into the batch. A rerun with the same checkpoint path now re-executes
  infra-errored trials and `SKIPPED` placeholders (`TIMEOUT` trials stay
  skipped — a timeout is an observation about the agent). The checkpoint file format changed to the
  identity envelope described above; old files remain readable.

### Fixed
- **`InfraError` docstring matched to behavior.** It previously claimed
  `OSError` and network `TimeoutError` were classified as infra; they never
  were. The docstring now describes the real (configurable) set and that
  the runner's own budget timeout is always `TIMEOUT`.
- **`compare_to_baseline_summary` no longer crashes at n=1.** The
  Welch-Satterthwaite degrees of freedom fell back to a division by zero
  when either side had a single sample.
- **Corrupt checkpoint files fail clearly.** An unreadable or unparseable
  checkpoint now raises `CheckpointError` with the offending path and a
  recovery hint (the CLI prints the error and exits 2 — the misconfigured-run contract) instead of an
  unhandled `JSONDecodeError`.
- **`RunnerConfig.fail_fast` is honored.** The field was previously accepted
  and silently ignored. When enabled, the first trial whose execution fails —
  final status `FAILED`, `INFRA_ERROR` (after `max_infra_retries` is
  exhausted), or `TIMEOUT` — stops new work from being scheduled. In-flight
  trials still run to completion; unstarted work items produce no trials at
  all, so pass rates, the baseline gate, and checkpoints only ever see
  trials that actually executed (a resume naturally runs the remainder).
  Trials that execute but fail grading, and teardown errors on otherwise
  successful trials, do not trip it. The runner logs how many work items
  were left unrun.

### Removed

- **`Task.max_retries`.** Dead configuration — the runner never read it.
  Retry policy is an execution concern and lives in
  `RunnerConfig.max_infra_retries`. Eval-set JSON containing the old field
  still loads; the value is ignored.

## [0.3.0] - 2026-06-10

Hardening release: the grading path now honors its own configuration, harness
failures are first-class signals, and long evaluations survive crashes.

### Added

- **Grader-crash tracking.** `Outcome.grader_error` marks outcomes synthesized
  from grader crashes; `Trial.has_grader_error` and
  `TrialBatch.grader_error_count`/`grader_error_rate` aggregate them, and
  reports carry the counts next to the existing infra-error stats. A spike
  here means the grading harness broke — not that the agent regressed.
- **Checkpoint/resume.** `RunnerConfig.checkpoint_path` and
  `checkpoint_interval` persist the batch atomically during long runs;
  re-running with the same path resumes, skipping completed trials. CLI:
  `--checkpoint`.
- **Progress reporting.** `RunnerConfig.progress_callback` is called with
  `(completed, total)` after each trial. CLI: `--progress` prints per-trial
  progress to stderr.
- **DecisionSpec wiring.** `EvaluationRunner(decision_spec=...)` stamps the
  spec onto every transcript that doesn't already carry one, so baselines
  record the reproducibility fingerprint of the run that produced them.
- **Token usage roll-up.** `TrialBatch.total_input_tokens` /
  `total_output_tokens` / `total_tokens`, mirrored on `ReportData`, for cost
  visibility without walking every transcript.
- **Quality infrastructure.** CLI end-to-end integration tests, a `Makefile`
  with a single `make verify` gate (lock check → lint → typecheck → tests +
  coverage), and a 90% coverage floor enforced in CI.

### Fixed

- **`LLMGrader` honors `GraderConfig`.** Each grading attempt is bounded by
  `timeout_seconds`, and transient failures — including malformed responses,
  which a fresh LLM call often fixes — retry per `retry_on_error` /
  `max_retries` with exponential backoff (new `retry_backoff_seconds` knob).
  These fields were previously accepted and silently ignored; a hung provider
  stalled the whole eval indefinitely.
- **`MemoryError` from graders propagates** (kill-switch) instead of being
  converted into bogus 0-score outcomes for the rest of the run.
- **CLI `--baseline-check` statistics.** The regression detector now receives
  one metric sample per trial instead of a single pre-aggregated dict,
  restoring the intended t-test over the sample distribution.

### Changed

- Generalized maintainer guidance and public docs for the open source library:
  removed private downstream project references, refreshed CI examples for the
  current CLI, and updated package constraints to the latest PyPI release
  (`0.2.0`).

## [0.2.0] - 2026-05-30

This release productizes the engine (install, understand, trust) and ships the
infra-noise differentiator. Two themes:

1. **Productization pre-v1.0** — turn the engine into something an external
   user can actually install, understand, and trust.
2. **Infra-noise differentiator** — implement Anthropic's "Quantifying
   infrastructure noise in agentic coding evals" recommendations
   (Feb 2026) as first-class framework features.

### Added

- **Human-eval calibration loop.** `sample_for_review()` and the
  `tracelens sample` command select trials for human review (strategies:
  `diverse`, `boundary`, `failures`, `random`) and emit a self-contained
  fill-in worksheet. The reviewer fills in grades, then `tracelens reconcile`
  (an alias for `calibrate`) pairs grader vs. human **per row** — carrying the
  grader outcome and `trial_id` in the worksheet, so no separate results file
  is needed and multi-run trials sharing a `task_id` stay distinct.
  `CalibrationAnalyzer.analyze_worksheet()` backs this. Bring-your-own human
  grades; no rating UI shipped.
- **`DecisionSpec.InfraConfig`.** Resource limits (CPU/memory guaranteed
  vs. hard-limit), time budget, concurrency level, sandbox provider, and
  harness version are now first-class experimental variables that feed
  into the `DecisionSpec` SHA-256 fingerprint. Observational fields
  (hostname, container ID, wall-clock start) are recorded but
  intentionally excluded from the fingerprint, so two runs with identical
  configs on different hosts collide to the same fingerprint. Based on
  Anthropic's recommendation that "resource configuration ... [be]
  documented and controlled with the same rigor as prompt format or
  sampling temperature."
- **`TrialStatus.INFRA_ERROR` + `InfraError` exception.** The runner now
  classifies `MemoryError`, `ConnectionError`, and adapter-raised
  `InfraError` as infrastructure failures, distinct from task-level
  `FAILED` status. `TrialBatch.infra_error_rate` and `infra_error_count`
  aggregate at the suite level.
- **`RegressionDetector.compare_with_specs()`.** New method that diffs
  the two runs' `InfraConfig`s. When they differ AND a regression's
  absolute delta is below the 3pp noise band (Anthropic's threshold),
  the regression is marked `within_noise_band=True` and filtered out
  of `blocking_regressions`. Default `should_block_ci()` honors that
  filter so CI doesn't gate on ambiguous noise.
- **`ReportGenerator` surfaces infra metrics.** `infra_error_rate`
  appears in markdown, CI summary, and HTML outputs when non-zero.
  Regression reports with an infra-config mismatch get an explicit
  warning block plus a "blocking / total" breakdown in the CI line.
- **Flagship benchmark pack** at `benchmarks/high-stakes-autonomous/`.
  Six tasks (safety GATE + resource-sensitive capability) plus a
  runnable script that reproduces Anthropic's finding end-to-end:
  same agent, same tasks, 100% → 50% pass rate purely from a memory
  budget change, correctly attributed to infrastructure rather than
  capability. Ships with a walkthrough README.
- **Three runnable examples** under `examples/`:
  - `http_agent_eval.py` — `HTTPAPIAdapter` + `JsonSchemaGrader` against
    an in-process stdlib HTTP server. Swap the URL for your real
    service.
  - `contract_eval.py` — `BehaviorContract.to_graders()` producing a
    mixed GATE/WARN/TRACK grader suite from one declaration.
  - `noise_aware_regression.py` — 98-line version of the infra-noise
    differentiator: two runs, different memory budgets, fingerprint
    mismatch, noise-aware regression report.
  - `human_eval_calibration.py` — reconcile a (deliberately miscalibrated)
    recorded grader against human grades; no API keys. Shows correlation,
    disagreement cases, and a tuning recommendation.
- **4-job GitHub Actions CI workflow** at `.github/workflows/ci.yml`:
  `core` (Python 3.11 / 3.12 / 3.13), `with-llm`, `examples-smoke`, and
  `lint`. Concurrency group cancels in-flight runs on re-push.
- **Tag-driven release workflow** at `.github/workflows/release.yml`.
  Versions come from `vX.Y.Z` git tags via `hatch-vcs`; the workflow builds,
  validates, checks the tag/version match, and publishes with PyPI trusted
  publishing.
- **OSS launch guidance**: `docs/releasing.md`, `docs/scenarios.md`, and
  `examples/README.md`.
- **Onboarding & recipe docs**: `docs/human-eval.md` (the full sample →
  grade → reconcile loop), `docs/baseline-regression-tutorial.md` (verified
  first-eval → CI-blocking baseline walkthrough), and
  `docs/evaluation-recipes.md` (the producer/evaluator/consumer pattern).
- **Curated public API** at `tracelens/*` (83 symbols). Top-level imports
  are the stable surface; submodules may move.
- **OSS hygiene files**: `LICENSE` (MIT), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`,
  `CHANGELOG.md` (this file), GitHub issue templates, and a pull
  request template.
- **Typed package marker**: `src/tracelens/py.typed` now ships in
  wheels so the `Typing :: Typed` classifier reflects the package
  artifact.

### Changed

- **pyproject.toml metadata** — real author + project URLs (repo,
  issues, changelog), richer classifiers, descriptive keywords. Moved
  `[project.urls]` below `dependencies` to fix a hatchling editable-
  install break.
- **Public installation docs** — first-path README, installation, and
  CI/CD guidance now describe public GitHub installs instead of private
  repository authentication.
- **CI type gate** — the workflow now runs strict mypy as a required
  job, matching the contributor verification checklist.
- **sdist contents** — maintainer-local `CLAUDE.md` is excluded from
  source distributions.
- **Ruff baseline** — applied autofixes to 67 pre-existing findings
  (unused imports, `datetime.utcnow()` → `datetime.now(UTC)`, f-string
  cleanup). Configured `ignore = ["E501", "UP042"]` for the remaining
  23 long lines and 7 `str, Enum` sites as deferred cleanup.
- **Factory policy change for LLM providers** — `create_provider()`
  now only supports `"in-memory"`; passing any other alias raises
  `ValueError` with guidance to subclass `LLMProvider` directly. This
  replaces the shipped LiteLLM wrapper with a documented subclassing
  pattern; tracelens no longer tries to own the provider integration
  layer.

### Removed

- **`WorkflowTask` / `WorkflowRunner` / `WorkflowAdapter`** (392 LOC +
  tests). Codex found that `workflow_runner.py:158-194` computed per-
  step grading outcomes then silently dropped them on the floor. No
  example, benchmark, CLI, or README referenced the workflow layer.
  Breaking change for the top-level API, but no known downstream user.
- **`LiteLLMProvider`** (`src/tracelens/llm/litellm_provider.py`, ~40
  LOC). A thin wrapper over `litellm.acompletion` with no distinct
  value over the vendor SDK. Users now subclass `LLMProvider`
  directly; the factory module docstring shows the pattern. The
  `[llm]` extra retains `openai` + `anthropic` as a convenience
  bundle, minus `litellm`.
- **`BehaviorContract.tool_param_constraints`** and
  **`BehaviorContract.max_cost_usd`**. Both declared, both unreferenced
  in `to_graders()`. Surface area for placeholders that never did
  anything.
- **Dead `if False` block + hardcoded `v0.1.0` strings** in
  `ReportGenerator.render_html`. Version string now sourced from
  `tracelens._version.__version__`.

### Fixed

- **Litellm-dependent tests FAIL vs. SKIP.** Guarded behind a
  `pytest.mark.skipif(find_spec("litellm") is None)` marker so the
  core-only CI job passes cleanly without the optional dep. Now
  obsolete after the LiteLLMProvider removal, but preserved the
  pattern for any future optional-dep-gated tests.
- **Degenerate regression samples no longer emit SciPy precision-loss
  warnings.** Zero-variance current samples now short-circuit the
  one-sample t-test and preserve the same significance outcome without
  leaking runtime warnings during Python 3.13 test runs.
- **Core installs no longer require the `[http]` extra at import time.**
  `HTTPAPIAdapter` still raises a targeted install hint when used
  without `httpx`, but `import tracelens` and `tracelens --help` now work
  from a base wheel install.
- **Metric grader tests no longer depend on a current event loop.** The sync
  test helpers in `test_validators.py` / `test_budgets.py` used
  `asyncio.get_event_loop().run_until_complete()`, which raised under
  `pytest-asyncio` 1.0 (it unsets the loop after async tests). Switched to
  `asyncio.run()`.

### Notes on backward compatibility

This is a `0.x` release, so breaking changes are allowed per the
version scheme noted at the top. The visible breaks are:

- `from tracelens import WorkflowTask` (and siblings) will now raise
  `ImportError`. No known downstream user imports these, but flagging
  the break in case you had one.
- `from tracelens.llm.litellm_provider import LiteLLMProvider` is gone;
  write a `LLMProvider` subclass against your vendor SDK instead.
  The pattern is in the `tracelens.llm.factory` module docstring.
- `BehaviorContract(tool_param_constraints=...)` or `max_cost_usd=...`
  will now raise `pydantic.ValidationError`. Neither had any effect
  before, so removing the kwarg is a no-op for real usage.

## [0.1.0] - Initial alpha

Initial scaffold shipped prior to this branch — `Task`, `Trial`,
`Grader`, `Transcript`, `AgentAdapter`, `EvaluationRunner`,
`BaselineManager`, `RegressionDetector`, `ReportGenerator`,
`DecisionSpec`, pass@k / pass^k / bootstrap CI statistics,
`tracelens run` CLI, etc. See git history before branch point for
the full list.
