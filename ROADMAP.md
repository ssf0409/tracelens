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

## Strategic Focus Areas

The issue tracker is the source of truth for current priority, ownership, and
acceptance criteria. This roadmap keeps the longer-lived direction stable.

### Direction And Positioning

TraceLens should make its scope obvious before users install it. The docs should
explain when to use TraceLens, when to choose adjacent tools, and how the project
decides whether a feature belongs in core or in a recipe.

### Fast First Evaluation

New users should get from an empty repo to a useful local eval quickly. The
project should continue improving scaffolding, CLI affordances, examples, and
CI templates until a first eval feels routine.

### Data Portability

TraceLens should accept common local eval data formats without pulling heavy
optional dependencies into core. JSON stays the native format; CSV and JSONL are
good core candidates; hosted datasets and framework-specific loaders should
start as optional integrations or recipes.

### Comparison And Regression Workflows

The statistical value in TraceLens should be visible from the CLI and CI, not
only from Python APIs. Baseline comparison, bootstrap significance, and
pass@k/pass^k reporting should be easy to run and hard to misread.

### Recipe Ecosystem

Integrations with LangChain, LangGraph, OpenAI SDK agents, OpenTelemetry traces,
RAG workflows, and public benchmark packs should live as examples first. Promote
shared code into core only after multiple downstream users prove the shape.

## Contributor On-Ramp

New contributors should start with scoped issues that have clear acceptance
criteria and a small review surface. Use the GitHub `good first issue` label for
the current queue; keep priority labels, milestones, and project-board fields in
GitHub rather than baking them into this file.

Before opening a PR, read [CONTRIBUTING.md](CONTRIBUTING.md), comment on the
issue with your intended approach, and run `make verify`.

## Release Hygiene

Every release should leave these channels agreeing with each other:

- `CHANGELOG.md` has a dated section for the version.
- The git tag exists on GitHub.
- PyPI shows the same latest version.
- GitHub Releases has a release for the tag.
- `mkdocs build --strict` exits cleanly with no project link or rendering
  warnings.

If any channel drifts, fix that before starting more feature work. Trust is a
feature.
