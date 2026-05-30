# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) —
but while `0.x`, minor bumps may contain breaking changes. Treat the
top-level `tracelens.*` imports as the stable surface; submodule paths may move.

## [Unreleased]

Nothing yet — queued for the next tag-driven release.

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
