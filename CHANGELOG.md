# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) —
but while `0.x`, minor bumps may contain breaking changes. Treat the
top-level `tracelens.*` imports as the stable surface; submodule paths may move.

## [Unreleased]

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

### Removed

- **`Task.max_retries`.** Dead configuration — the runner never read it.
  Retry policy is an execution concern and lives in
  `RunnerConfig.max_infra_retries`. Eval-set JSON containing the old field
  still loads; the value is ignored.

### Fixed

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
