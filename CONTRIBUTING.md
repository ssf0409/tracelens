# Contributing to TraceLens

Thanks for your interest in improving TraceLens. This project aims to be a *trustworthy* evaluation framework for autonomous AI agents, so contributions that improve reproducibility, reliability, or noise-awareness are especially welcome.

TraceLens follows a **thin core, rich recipes** direction. The framework owns
evaluation primitives, statistics, baselines, reports, and calibration. Domain
truth, provider-specific integrations, hosted dashboards, and rollout policy
belong in downstream projects or examples. Read [ROADMAP.md](ROADMAP.md) before
proposing a broad new abstraction.

## Ways to contribute

- **Bug reports** — open an issue with a minimal repro (the smaller, the better).
- **Feature requests** — describe the agent/eval scenario you're trying to support before proposing an API.
- **Pull requests** — see below.
- **Benchmark contributions** — new task packs for public benchmarks under `benchmarks/` are very welcome.

## First PR Path

1. Pick a scoped issue from the GitHub `good first issue` label. Three that
   are genuinely small, each with concrete acceptance criteria in the issue:
   - [#74](https://github.com/ssf0409/tracelens/issues/74) — stop the
     hello-world test rewriting the checked-in sample reports (one example,
     one test, one doc note).
   - [#75](https://github.com/ssf0409/tracelens/issues/75) — `tracelens
     report --format ci`, re-rendering the one-line CI summary from a saved
     results file (one CLI branch, tests, one doc paragraph).
   - [#76](https://github.com/ssf0409/tracelens/issues/76) — show
     `provenance_version` in the adapter and grader examples and the
     scaffold templates (docs and two template strings).

   Statistical design (estimators, comparison semantics, gate policy) is
   owner-level work and is never labelled a first issue; it starts from the
   [statistical contract](docs/statistical-contract.md) and a maintainer
   discussion.
2. Comment on the issue with the approach you plan to take.
3. Keep the PR to one behavior change. If you discover a larger architecture
   cleanup, call it out in the PR body instead of expanding silently.
4. Add or update tests for every behavior change.
5. Run `make verify` before requesting review.

## Where things live

The full tree is in [CLAUDE.md](CLAUDE.md); the short version:

- `core/` — `Task`, `Transcript`, `Trial`, `Outcome`, `DecisionSpec`, and run
  provenance.
- `execution/` — the runner (concurrency, timeouts, retries, checkpoints) and
  adapters; `loaders/` — JSON, JSONL, and CSV task loaders in core, Hugging
  Face behind the `[datasets]` extra.
- `statistics/` — pass@k, pass^k, bootstrap intervals, and the paired run
  comparison; every estimator follows
  [docs/statistical-contract.md](docs/statistical-contract.md), which changes
  in the same PR as the estimator.
- `baselines/` and `reporting/gate.py` — baselines and the CI gate decision;
  `reporting/` — Markdown, JSON, HTML, the CI summary, and failure inspection.
- `calibration/` — human-review sampling and grader-versus-human
  reconciliation.
- `cli/` — `run`, `report`, `compare`, `inspect`, `sample`, `reconcile`,
  `init`, and the `tracelens.yaml` config.

## Development setup

```bash
git clone https://github.com/ssf0409/tracelens.git
cd tracelens

# Recommended: uv
uv venv
uv pip install -e ".[dev]"

# Or plain pip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Optional dependency groups

The core install (`pydantic`, `numpy`, `scipy`, `jsonschema`, `pyyaml`) is all
the CLI needs, including `tracelens.yaml`, the JSON/JSONL/CSV loaders, and the
calibration commands (`sample`, `reconcile`). Extras:

- `.[llm]` — Convenience bundle of `openai` + `anthropic` SDKs for users who subclass `LLMProvider` directly. Never required by the tests.
- `.[http]` — `httpx`. Required for `HTTPAPIAdapter`.
- `.[datasets]` — `datasets`, for the optional Hugging Face task loader. A separate CI job runs the suite with it installed.
- `.[dev]` — dev tools (pytest, pytest-asyncio, pytest-cov, ruff, mypy, httpx, type stubs).
- `.[docs]` — MkDocs Material and mkdocstrings for the documentation site.

### Running the verification gate

```bash
make verify   # uv lock --check -> ruff -> mypy (strict) -> pytest with the 90 % coverage floor
```

That is exactly what CI runs. The individual commands are in
[CLAUDE.md](CLAUDE.md) under "Testing". Documentation changes must also pass
`make docs` (`mkdocs build --strict`).

For changes that affect packaging, console scripts, public imports, examples,
or downstream dependency behavior, also run the relevant environment checks in
[docs/contributor-testing.md](docs/contributor-testing.md). In particular,
wheel smoke tests should use a clean virtual environment, not the editable dev
checkout.

## Pull request guidelines

1. **One change per PR.** If your branch touches the adapter layer *and* the statistics layer, split it.
2. **Write tests first** (TDD). Failing tests go in the same commit as the fix/feature.
3. **Don't mock at system boundaries.** Tests that pretend the HTTP adapter works without actually exercising it create false confidence.
4. **Update public API exports deliberately.** Adding something to `src/tracelens/__init__.py` is a stability promise. If you're unsure whether a symbol belongs in the public surface, leave it at the submodule path.
5. **Document "why" in the PR body.** Commit messages should explain the user-visible behavior change; PR bodies should explain the motivation (what problem does this solve? what alternatives were considered?).
6. **If you modify regression / baseline logic, add a backwards-compat note** to `CHANGELOG.md`. Baselines are a stability boundary.
7. **Prefer recipes before core abstractions.** If only one downstream project
   needs the behavior, start with docs or examples. Promote it into core only
   after the shape is proven.

## Commit style

- `feat:` / `fix:` / `docs:` / `test:` / `refactor:` / `chore:` prefixes (Conventional Commits, lenient).
- Present tense, imperative mood. ("add X", not "added X".)
- Keep commits small and logically atomic so `git blame` tells a useful story.

## Code style

- 4 spaces, no tabs.
- Type hints on every function (including private helpers).
- No wildcard imports except in `__init__.py`.
- Keep line length ≤ 100 (ruff-enforced).
- Module-level imports only; no imports inside functions.

## Design principles

These guide reviews; deviations should be justified in the PR description:

1. **Grade outcomes, not paths.** If a new grader needs to inspect intermediate steps, think hard about whether you can grade the final artifact instead.
2. **Explicit provenance.** Anything declared to affect agent behavior — model, prompt, tool availability, *infrastructure* — belongs in `DecisionSpec`; what was measured (task content, graders, settings) is recorded on every run as provenance. Both are evidence for attributing a change, never proof of cause or of identical execution, and the docs must say so.
3. **Policies, not booleans.** Graders carry a policy (GATE / WARN / TRACK), not a hard-coded `is_critical` flag.
4. **No silent fallbacks.** If a dependency is missing or a call fails, raise loudly with context.

## Releasing (maintainers only)

Releases are tag-driven and the tag is created for you:

1. Keep `CHANGELOG.md` current: every user-visible change lands under
   `[Unreleased]` in the same pull request as the code.
2. Run the "Release prepare" workflow with the version. It moves
   `[Unreleased]` into a dated section and opens a `release: vX.Y.Z` pull
   request with the rendered notes.
3. Review and merge that pull request (any merge method). The "Release tag"
   workflow tags the commit that lands on `main` and the release workflow
   publishes to PyPI and creates the GitHub Release from the changelog
   section.

The full checklist, the verification commands, the manual fallback, and what
to do when a step fails are in [docs/releasing.md](docs/releasing.md).

## Questions?

Open a GitHub Discussion or issue. For security reports, see [SECURITY.md](SECURITY.md).
