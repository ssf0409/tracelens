# Contributor Testing Guide

This guide defines the development and pre-release environments contributors
should use before a change reaches PyPI.

For TraceLens, "production" means the published PyPI package plus the public
GitHub documentation that downstream projects install from. Normal contributors
do not publish releases; maintainers publish through the tag-driven release
workflow.

## Environment Ladder

Use the smallest environment that catches the risk in your change. Move down
the ladder as the blast radius grows.

| Environment | Use When | Required Commands |
|-------------|----------|-------------------|
| Editable dev checkout | Changing source, tests, or docs locally. | `uv sync --extra dev --extra http` |
| Local verification gate | Before every PR. | `uv lock --check`, `uv run --frozen --extra dev pytest -q`, `uv run --frozen --extra dev ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous`, `uv run --frozen --extra dev mypy src/tracelens/` |
| Built artifact smoke | Changing packaging, CLI, public imports, README, or release metadata. | Build wheel/sdist, install the wheel into a clean venv, run import and CLI smoke tests. |
| Downstream integration smoke | Changing public APIs, dependency metadata, adapters, baselines, or statistics. | Install the built wheel into a small downstream fixture or real downstream project and run its TraceLens-facing tests. |
| PyPI release | Maintainer-only final publish path. | Push `vX.Y.Z`; GitHub Actions publishes with PyPI trusted publishing. |

## Local Editable Environment

```bash
git clone https://github.com/ssf0409/tracelens.git
cd tracelens
uv sync --extra dev --extra http
```

Use `[llm]` only when you are working on examples or docs that need OpenAI or
Anthropic SDK imports:

```bash
uv sync --extra dev --extra http --extra llm
```

The core test suite must not require live LLM credentials.

## PR Verification Gate

Run this before opening or updating a PR:

```bash
uv lock --check
uv run --frozen --extra dev pytest -q
uv run --frozen --extra dev ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous
uv run --frozen --extra dev mypy src/tracelens/
```

If your change touches examples, run the changed examples directly. At minimum:

```bash
uv run --frozen python examples/hello_world.py
uv run --frozen python examples/contract_eval.py
uv run --frozen --extra http python examples/http_agent_eval.py
uv run --frozen python examples/noise_aware_regression.py
```

## Built Artifact Smoke

Use this when a change could affect what users install from PyPI:

```bash
uv build --sdist --wheel --out-dir /tmp/tracelens-dist
python -m venv /tmp/tracelens-wheel-smoke
/tmp/tracelens-wheel-smoke/bin/python -m pip install /tmp/tracelens-dist/tracelens-*.whl
/tmp/tracelens-wheel-smoke/bin/python -c "import tracelens; print(tracelens.__version__)"
/tmp/tracelens-wheel-smoke/bin/tracelens --help
```

This catches missing package files, broken console scripts, stale version
metadata, and import errors that editable installs can hide.

## User-Journey Test

`tests/journey/test_user_journey.py` runs the documented workflow as a user
would, as real `tracelens` processes in a scratch project: an existing
project, `tracelens init`, `run --config`, baselines stored with the README
snippet, the gate enabled in `tracelens.yaml`, an intentional regression that
blocks, `inspect` explaining it, `compare` calling it a regression, a
`--task-id` rerun, an infra outage and a grader crash made unevaluable and
told apart, malformed input and a bad config as usage errors,
checkpoint/resume re-executing nothing, the suite passing again, and
`report` and `sample` reading the artifacts. Every step checks the exit code
and the decision persisted in the files.

`make verify` runs it against the editable install. CI (the `user journey
(built wheel)` job) runs it against the console script of a freshly built
wheel in a clean environment, the boundary no other test crosses. To do the
same locally:

```bash
uv build --wheel --out-dir /tmp/tracelens-dist
uv venv /tmp/tracelens-journey
uv pip install --python /tmp/tracelens-journey/bin/python \
  /tmp/tracelens-dist/tracelens-*.whl pytest pytest-asyncio
TRACELENS_CLI=/tmp/tracelens-journey/bin/tracelens \
  /tmp/tracelens-journey/bin/python -m pytest tests/journey -q
```

`TRACELENS_CLI` names the executable (a path, or a command line) the journey
drives; unset, it falls back to `python -m tracelens.cli.main` from the
interpreter running pytest. When a documented command changes, extend the
journey rather than adding a separate smoke test.

## Downstream Integration Smoke

For public API or dependency changes, test from the perspective of a downstream
project:

```toml
[project]
dependencies = [
    "tracelens>=0.3.0",
]
```

Before a release, install the locally built wheel into the downstream
environment instead of depending on an unpublished Git commit:

```bash
uv build --wheel --out-dir /tmp/tracelens-dist
cd /path/to/downstream-project
uv pip install /tmp/tracelens-dist/tracelens-*.whl
uv run python -c "import tracelens; print(tracelens.__version__)"
```

Do not rely on a globally installed editable checkout for downstream smoke
tests; it can mask packaging and dependency problems.

## TestPyPI

TestPyPI is optional. Use it only when changing the publishing workflow itself,
trusted publishing configuration, or package metadata that cannot be validated
with a local wheel. For normal source, docs, examples, and API changes, the
built artifact smoke is faster and closer to what maintainers need before
tagging a release.

## Contributor Boundaries

- The hello-world integration test writes reports to pytest's temporary directory. To refresh the
  checked-in samples intentionally, run `python examples/hello_world.py` without `--output-dir`.
- Do not push release tags from contributor branches.
- Do not add real API keys, tokens, private URLs, or local machine paths to
  tests or docs.
- Keep LLM-dependent tests opt-in and clearly documented.
- Update examples or docs when changing a public API.
- Include the verification commands you ran in the PR body.
