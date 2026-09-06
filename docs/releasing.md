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
- The release workflow builds the package, verifies the tag matches the built
  version, renders the release notes from the changelog's dated section,
  publishes to PyPI using trusted publishing, and then creates the GitHub
  Release for the tag with those notes and the built files attached.
- Nothing is published unless the build, the version check, and the notes
  succeed first, and no GitHub Release is created unless publication
  succeeded. Tags whose version is not a final `X.Y.Z` (`rc`, `a`, `b`,
  `dev`, `post`) are marked as pre-releases on GitHub.

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

   The release workflow takes the GitHub Release notes from exactly this
   section and fails, before publishing anything, if it is missing or empty.
   Preview what it will render at any time:

   ```bash
   python scripts/release_notes.py --version X.Y.Z
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

6. Watch the GitHub Actions release workflow. Its three jobs run in order:
   `build, verify, render notes` (the notes preview appears in the job
   summary), `publish to PyPI` (in the `release` environment), and
   `create GitHub Release`.

7. Verify that every public channel tells the same story:

   ```bash
   git ls-remote --tags origin "vX.Y.Z"           # the tag exists on GitHub
   python -m pip index versions tracelens          # PyPI lists X.Y.Z
   gh release view vX.Y.Z --json isPrerelease,assets,body   # the GitHub Release exists,
                                                   # is (not) a pre-release, has the wheel
                                                   # and sdist, and its body is the
                                                   # changelog section
   python scripts/release_notes.py --version X.Y.Z # same notes locally
   mkdocs build --strict                           # docs still build
   ```

8. After PyPI publish completes, smoke test from a clean environment:

   ```bash
   python -m venv /tmp/tracelens-release-smoke
   /tmp/tracelens-release-smoke/bin/python -m pip install tracelens
   /tmp/tracelens-release-smoke/bin/tracelens --help
   ```

## If A Release Fails

- **Build, version check, or release notes fail.** Nothing was published.
  Fix the cause (usually a missing changelog section or a tag that does not
  match the version), delete the tag locally and remotely, and push it again.
- **PyPI publication fails.** No GitHub Release is created. PyPI versions are
  immutable, so if the failure happened *after* upload, do not retag; if it
  happened before, fix and re-run the workflow.
- **PyPI succeeded but the GitHub Release step failed.** Re-run the workflow
  (either "Re-run failed jobs" or "Re-run all jobs"): publishing is skipped
  for files PyPI already has (`skip-existing`), and the release step updates
  an existing release instead of failing on "already exists". Re-running
  never uploads a different file under an already-published version.
- **Something else needs checking first.** Use "Run workflow"
  (`workflow_dispatch`) on a branch or a tag: it builds, verifies, and shows
  the rendered notes in the job summary without publishing anything.

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
