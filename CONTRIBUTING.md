# Contributing to TraceLens

Thanks for your interest in improving TraceLens. This project aims to be a *trustworthy* evaluation framework for autonomous AI agents, so contributions that improve reproducibility, reliability, or noise-awareness are especially welcome.

## Ways to contribute

- **Bug reports** — open an issue with a minimal repro (the smaller, the better).
- **Feature requests** — describe the agent/eval scenario you're trying to support before proposing an API.
- **Pull requests** — see below.
- **Benchmark contributions** — new task packs for public benchmarks under `benchmarks/` are very welcome.

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

- `.[llm]` — Convenience bundle of `openai` + `anthropic` SDKs for users who subclass `LLMProvider` directly. Not required by the core test suite.
- `.[http]` — `httpx`. Required for `HTTPAPIAdapter`.
- `.[dev]` — dev tools (pytest, ruff, mypy, httpx).

The test suite does not require the `[llm]` extra — tracelens no longer ships a built-in third-party provider wrapper, so there are no optional-dep-gated tests.

### Running the verification gate

```bash
pytest -q                          # all tests
ruff check src/ tests/             # lint
mypy src/tracelens/                 # type check (strict mode)
```

All three must pass before opening a PR.

## Pull request guidelines

1. **One change per PR.** If your branch touches the adapter layer *and* the statistics layer, split it.
2. **Write tests first** (TDD). Failing tests go in the same commit as the fix/feature.
3. **Don't mock at system boundaries.** Tests that pretend the HTTP adapter works without actually exercising it create false confidence.
4. **Update public API exports deliberately.** Adding something to `src/tracelens/__init__.py` is a stability promise. If you're unsure whether a symbol belongs in the public surface, leave it at the submodule path.
5. **Document "why" in the PR body.** Commit messages should explain the user-visible behavior change; PR bodies should explain the motivation (what problem does this solve? what alternatives were considered?).
6. **If you modify regression / baseline logic, add a backwards-compat note** to `CHANGELOG.md`. Baselines are a stability boundary.

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
2. **Explicit reproducibility.** Anything that affects agent behavior — model, prompt, tool availability, *infrastructure* — belongs in `DecisionSpec`.
3. **Policies, not booleans.** Graders carry a policy (GATE / WARN / TRACK), not a hard-coded `is_critical` flag.
4. **No silent fallbacks.** If a dependency is missing or a call fails, raise loudly with context.

## Releasing (maintainers only)

TraceLens uses tag-driven releases. Do not commit a version bump just to
release. Package versions are generated from git tags by `hatch-vcs`.

1. Move `[Unreleased]` notes to a new dated section in `CHANGELOG.md`.
2. Run the full verification gate:
   `uv lock --check`, `uv run --frozen pytest -q`,
   `uv run --frozen ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous`,
   `uv run --frozen --extra dev mypy src/tracelens/`, and
   `uv build --sdist --wheel`.
3. Commit the release notes.
4. Push a tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
5. Watch `.github/workflows/release.yml`.

See [docs/releasing.md](docs/releasing.md) for the PyPI trusted publishing
setup and release checklist.

## Questions?

Open a GitHub Discussion or issue. For security reports, see [SECURITY.md](SECURITY.md).
