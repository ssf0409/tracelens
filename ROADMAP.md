# TraceLens Roadmap

TraceLens is a thin-core evaluation and regression-testing framework for AI
agents. The core package should stay small, local, inspectable, and CI-ready.
Domain truth, rollout policy, hosted dashboards, and framework-specific agent
integration belong in downstream projects or examples.

## Product Doctrine

**Thin core, rich recipes.**

TraceLens should own the reusable evaluation primitives:

- Task, transcript, trial, outcome, and grader models.
- Agent adapters and execution orchestration.
- pass@k, pass^k, bootstrap confidence intervals, and regression detection.
- Baselines, reproducibility fingerprints, and CI summaries.
- Human-eval calibration worksheets and reconciliation.

TraceLens should not become:

- A hosted tracing or observability backend.
- A prompt playground or prompt-management platform.
- A broad provider SDK wrapper.
- A domain-specific benchmark suite hidden inside the core package.
- A RAG metrics framework unless that need is proven by multiple real users.

When a request could be solved by either a core feature or a recipe, prefer the
recipe until at least two downstream projects need the same abstraction.

## Near-Term Priorities

### P0: Make Direction Legible

- Publish this roadmap and keep it linked from README and contributing docs.
- Add a positioning page that explains when to choose TraceLens versus adjacent
  tools.
- Keep GitHub Releases, PyPI, docs, and changelog in sync for every release.
- Give contributors a small set of well-scoped first issues.

### P1: Reduce Time To First Useful Eval

- `tracelens init`: scaffold a runnable eval directory with tasks, adapter,
  grader, README, and CI template.
- CSV and JSONL task loaders: let users bring existing eval data without
  converting it by hand. Keep optional HuggingFace support as a separate,
  optional-dependency follow-up.
- `tracelens compare`: compare two saved runs with bootstrap significance from
  the CLI.

### P2: Recipes, Not Core Dependencies

- Integration cookbook for LangChain, LangGraph, and OpenAI SDK agents as
  examples, with lazy imports and no new core dependencies.
- OpenTelemetry trace import example that turns spans into TraceLens
  transcripts for regression reporting.
- More public benchmark packs that demonstrate specific evaluation patterns
  without making those domains part of the framework.

## Current Good First Issues

These are scoped so a new contributor can make a useful change without owning a
large redesign:

| Issue | Why it matters | Expected scope |
|---|---|---|
| [#25 `tracelens init`](https://github.com/ssf0409/tracelens/issues/25) | Fast first success for new users. | Add one CLI command, template files, and CLI tests. |
| [#28 `tracelens compare`](https://github.com/ssf0409/tracelens/issues/28) | Exposes existing statistics as a daily workflow. | Wrap existing comparison helpers; no new statistical model. |
| [#29 positioning page](https://github.com/ssf0409/tracelens/issues/29) | Helps users self-select before investing time. | Docs-only, fair comparison by capability. |

If you are new to the project, start with [CONTRIBUTING.md](CONTRIBUTING.md) and
run `make verify` before opening a PR.

[#27 CSV/JSONL task loaders](https://github.com/ssf0409/tracelens/issues/27)
is also open, but it is already in review through PR #31 and needs a narrower
CSV/JSONL-first slice before it becomes a good first PR again.

## Release Hygiene

Every release should leave these channels agreeing with each other:

- `CHANGELOG.md` has a dated section for the version.
- The git tag exists on GitHub.
- PyPI shows the same latest version.
- GitHub Releases has a release for the tag.
- The docs build without warnings.

If any channel drifts, fix that before starting more feature work. Trust is a
feature.
