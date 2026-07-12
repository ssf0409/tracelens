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

TraceLens already has a PyPI project and trusted publishing configured. Re-run
this section only if the repository, workflow name, or PyPI ownership changes.

1. Confirm the package metadata:

   ```bash
   python -m pip index versions tracelens
   ```

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
   ## [X.Y.Z] - YYYY-MM-DD
   ```

2. Ensure the verification gate is green:

   ```bash
   uv lock --check
   uv run --frozen --extra dev pytest -q
   uv run --frozen --extra dev ruff check src/ tests/ examples/ benchmarks/high-stakes-autonomous
   uv run --frozen --extra dev mypy src/tracelens/
   uv build --sdist --wheel
   ```

3. Run the release-relevant environment checks from
   [Contributor Testing](contributor-testing.md), especially the clean wheel
   smoke when packaging, CLI, README, public imports, or dependency metadata
   changed.

4. Commit the release notes.

5. Create and push the tag:

   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

6. Watch the GitHub Actions release workflow.

7. Create the matching GitHub Release for the tag. Use the changelog section as
   the release notes so GitHub and PyPI tell the same story:

   ```bash
   gh release create vX.Y.Z --title "vX.Y.Z" --notes-file /tmp/tracelens-vX.Y.Z.md
   ```

8. Verify all public release channels agree:

   ```bash
   python -m pip index versions tracelens
   gh release list --limit 5
   git ls-remote --tags origin "vX.Y.Z"
   ```

9. After PyPI publish completes, smoke test from a clean environment:

   ```bash
   python -m venv /tmp/tracelens-release-smoke
   /tmp/tracelens-release-smoke/bin/python -m pip install tracelens
   /tmp/tracelens-release-smoke/bin/tracelens --help
   ```

## Dependency Guidance

Downstream projects should depend on TraceLens from PyPI:

```toml
dependencies = [
    "tracelens>=0.3.0",
]
```

Public GitHub or PyPI dependencies do not need a CI secret. A secret is only
needed when a downstream CI job checks out or installs a private repository.

For local pre-release checks, prefer the built-wheel and downstream smoke
guidance in [Contributor Testing](contributor-testing.md). TestPyPI is optional
and mainly useful when changing the publishing workflow itself.
