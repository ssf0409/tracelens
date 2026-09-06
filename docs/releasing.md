# Releasing TraceLens

TraceLens uses tag-driven releases. The package version comes from the git
tag, and CI publishes only from a release tag: the release pipeline creates
one when a release pull request is merged, or a maintainer pushes one by hand.

This avoids CI-generated version commits, release loops, and PyPI's immutable
version constraint.

## Release Model

- Every pull request and every `main` commit runs tests, lint, typecheck, and
  package build validation.
- A tag named `vX.Y.Z` builds package version `X.Y.Z`.
- Pushing that tag triggers `.github/workflows/release.yml`.
- The "Release prepare" workflow turns the changelog's `[Unreleased]` section
  into a dated section and opens a release pull request; merging it tags the
  merge commit (the "Release tag" workflow).
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

The default path is two clicks and one review.

1. **Run the "Release prepare" workflow** (Actions → Release prepare → Run
   workflow) with the version, for example `0.5.0`, and an optional
   one-paragraph summary. It checks that the version is well-formed, not
   older than the latest tag, and unused; moves the changelog's
   `[Unreleased]` entries into `## [0.5.0] - <today>` (refusing an empty
   section); pushes `release/v0.5.0`; opens the pull request
   `release: v0.5.0` with the rendered notes as its description; and starts
   CI on the branch. Preview the notes locally at any time:

   ```bash
   python scripts/prepare_release.py --version 0.5.0 --check
   python scripts/release_notes.py --version 0.5.0   # after the section exists
   ```

2. **Review and merge the release pull request.** Edit `CHANGELOG.md` on the
   branch if the wording needs work; the pull request description is a
   preview, the changelog section is the source. Its checks are the `ci.yml`
   run the workflow started on the branch (pull requests opened by a
   workflow do not trigger `pull_request` workflows themselves). Merge it
   with any merge method: the "Release tag" workflow reads the version from
   the commit that lands on `main` (`release: v0.5.0` after a squash or
   rebase merge, the `release/v0.5.0` branch name in a merge commit), tags
   that commit `v0.5.0`, and starts the release workflow with `publish=true`,
   which publishes to PyPI (the `release` environment and any reviewer
   required there still apply) and creates the GitHub Release.

3. **Verify that every public channel tells the same story:**

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

4. **Smoke test from a clean environment** once PyPI lists the version:

   ```bash
   python -m venv /tmp/tracelens-release-smoke
   /tmp/tracelens-release-smoke/bin/python -m pip install tracelens
   /tmp/tracelens-release-smoke/bin/tracelens --help
   ```

Nothing releases on an ordinary merge to `main`: only the merge of a
`release/vX.Y.Z` pull request is tagged (the workflow also checks that the
changelog has the dated section and that the tag is new), and only a tag is
ever published.

### Manual fallback

If the workflows are unavailable, the tag-driven path still works on its own:

1. Move the `[Unreleased]` entries into `## [X.Y.Z] - YYYY-MM-DD` (or run
   `python scripts/prepare_release.py --version X.Y.Z`) and commit.
2. Run the verification gate (`make verify`, `make docs`) and, for packaging
   or dependency changes, the built-wheel smoke in
   [Contributor Testing](contributor-testing.md).
3. `git tag vX.Y.Z && git push origin vX.Y.Z`. The pushed tag triggers the
   release workflow directly, exactly as the automatic path does.

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
  (`workflow_dispatch`) on the release workflow with `publish` left off, on a
  branch or a tag: it builds, verifies, and shows the rendered notes in the
  job summary without publishing anything.
- **A release pull request must be abandoned.** Close it and delete the
  `release/vX.Y.Z` branch; nothing was tagged or published. Run "Release
  prepare" again later, with the same version if it is still right.
- **The tag was created but the release workflow never ran** (the dispatch
  in "Release tag" failed). Re-run the "Release tag" job: it keeps a tag that
  already points at the release commit and only dispatches again. Or run the
  release workflow by hand on the tag with `publish=true`; both are safe to
  repeat.

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
