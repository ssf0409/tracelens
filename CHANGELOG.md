# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) —
but while `0.x`, minor bumps may contain breaking changes. Treat the
top-level `eval_kit.*` imports as the stable surface; submodule paths may move.

## [Unreleased]

Everything under this heading is shipped on branch
`feat/eval-kit-feature-gaps` (15 commits ahead of `main`) and is queued for
the next release. Two themes:

1. **Productization pre-v1.0** — turn the engine into something an external
   user can actually install, understand, and trust.
2. **Infra-noise differentiator** — implement Anthropic's "Quantifying
   infrastructure noise in agentic coding evals" recommendations
   (Feb 2026) as first-class framework features.

### Added

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
- **4-job GitHub Actions CI workflow** at `.github/workflows/ci.yml`:
  `core` (Python 3.11 / 3.12 / 3.13), `with-llm`, `examples-smoke`, and
  `lint`. Concurrency group cancels in-flight runs on re-push.
- **Curated public API** at `eval_kit/*` (83 symbols). Top-level imports
  are the stable surface; submodules may move.
- **OSS hygiene files**: `LICENSE` (MIT), `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`,
  `CHANGELOG.md` (this file).

### Changed

- **pyproject.toml metadata** — real author + project URLs (repo,
  issues, changelog), richer classifiers, descriptive keywords. Moved
  `[project.urls]` below `dependencies` to fix a hatchling editable-
  install break.
- **Ruff baseline** — applied autofixes to 67 pre-existing findings
  (unused imports, `datetime.utcnow()` → `datetime.now(UTC)`, f-string
  cleanup). Configured `ignore = ["E501", "UP042"]` for the remaining
  23 long lines and 7 `str, Enum` sites as deferred cleanup.
- **Factory policy change for LLM providers** — `create_provider()`
  now only supports `"in-memory"`; passing any other alias raises
  `ValueError` with guidance to subclass `LLMProvider` directly. This
  replaces the shipped LiteLLM wrapper with a documented subclassing
  pattern; eval-kit no longer tries to own the provider integration
  layer.

### Removed

- **`WorkflowTask` / `WorkflowRunner` / `WorkflowAdapter`** (392 LOC +
  tests). Codex found that `workflow_runner.py:158-194` computed per-
  step grading outcomes then silently dropped them on the floor. No
  example, benchmark, CLI, or README referenced the workflow layer.
  Breaking change for the top-level API, but no known downstream user.
- **`LiteLLMProvider`** (`src/eval_kit/llm/litellm_provider.py`, ~40
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
  `eval_kit._version.__version__`.

### Fixed

- **Litellm-dependent tests FAIL vs. SKIP.** Guarded behind a
  `pytest.mark.skipif(find_spec("litellm") is None)` marker so the
  core-only CI job passes cleanly without the optional dep. Now
  obsolete after the LiteLLMProvider removal, but preserved the
  pattern for any future optional-dep-gated tests.

### Notes on backward compatibility

This is a `0.x` release, so breaking changes are allowed per the
version scheme noted at the top. The visible breaks are:

- `from eval_kit import WorkflowTask` (and siblings) will now raise
  `ImportError`. No known downstream user imports these, but flagging
  the break in case you had one.
- `from eval_kit.llm.litellm_provider import LiteLLMProvider` is gone;
  write a `LLMProvider` subclass against your vendor SDK instead.
  The pattern is in the `eval_kit.llm.factory` module docstring.
- `BehaviorContract(tool_param_constraints=...)` or `max_cost_usd=...`
  will now raise `pydantic.ValidationError`. Neither had any effect
  before, so removing the kwarg is a no-op for real usage.

## [0.1.0] - Initial alpha

Initial scaffold shipped prior to this branch — `Task`, `Trial`,
`Grader`, `Transcript`, `AgentAdapter`, `EvaluationRunner`,
`BaselineManager`, `RegressionDetector`, `ReportGenerator`,
`DecisionSpec`, pass@k / pass^k / bootstrap CI statistics,
`eval-kit run` CLI, etc. See git history before branch point for
the full list.
