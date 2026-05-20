# Releasing TraceLens

TraceLens uses tag-driven releases. The package version comes from the git
tag, and CI publishes only when a maintainer pushes a release tag.

This avoids CI-generated version commits, release loops, and PyPI's immutable
version constraint.

## Release Model

- Every pull request and every `main` commit runs tests, lint, typecheck, and
  package build validation.
- A tag named `vX.Y.Z` builds package version `X.Y.Z`.
- Pushing that tag triggers `.github/workflows/release.yml`.
- The release workflow publishes to PyPI using trusted publishing.

## One-Time PyPI Setup

Before the first release, create the PyPI project and configure trusted
publishing:

1. Confirm the package name is available:

   ```bash
   python -m pip index versions tracelens
   ```

   A "No matching distribution found" response means the name is currently
   unclaimed on PyPI.

2. In PyPI, add a trusted publisher for this repository:

   - Owner: `ssf0409`
   - Repository: `tracelens`
   - Workflow: `release.yml`
   - Environment: `release`

3. In GitHub, create the `release` environment under repository settings.
   Add required reviewers if you want a manual approval gate before publishing.

No PyPI API token is required when trusted publishing is configured correctly.

## Cut A Release

1. Move changelog entries from `[Unreleased]` to a dated version section:

   ```markdown
   ## [0.1.0] - 2026-05-20
   ```

2. Ensure the verification gate is green:

   ```bash
   uv lock --check
   uv run --frozen pytest -q
   uv run --frozen ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous
   uv run --frozen --extra dev mypy src/tracelens/
   uv build --sdist --wheel
   ```

3. Commit the release notes.

4. Create and push the tag:

   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

5. Watch the GitHub Actions release workflow.

6. After PyPI publish completes, smoke test from a clean environment:

   ```bash
   python -m venv /tmp/tracelens-release-smoke
   /tmp/tracelens-release-smoke/bin/python -m pip install tracelens
   /tmp/tracelens-release-smoke/bin/tracelens --help
   ```

## Dependency Guidance

Before the first PyPI release, downstream projects can depend on GitHub:

```toml
dependencies = [
    "tracelens @ git+https://github.com/ssf0409/tracelens.git",
]
```

After the first PyPI release, prefer normal package constraints:

```toml
dependencies = [
    "tracelens>=0.1.0",
]
```

Public GitHub or PyPI dependencies do not need a CI secret. A secret is only
needed when a downstream CI job checks out or installs a private repository.
