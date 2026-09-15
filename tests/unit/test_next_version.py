"""Tests for scripts/next_version.py, which the release-auto workflow runs."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "next_version.py"
REPO = SCRIPT.parents[1]
spec = importlib.util.spec_from_file_location("next_version", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules["next_version"] = module  # dataclasses resolve the module by name
spec.loader.exec_module(module)
ReleaseError = module.ReleaseError
decide = module.decide
inferred_bump = module.inferred_bump
latest_tag = module.latest_tag
parse_title = module.parse_title

TAGS = ["v0.3.0", "v0.4.0", "v0.5.0", "not-a-version", "v0.5.0rc1"]


def changelog(unreleased: str) -> str:
    return (
        "# Changelog\n\nIntro.\n\n## [Unreleased]\n\n"
        + unreleased
        + "\n## [0.5.0] - 2026-09-06\n\n- Older.\n"
    )


FIXES = "### Fixed\n\n- A bug. (#12)\n"
FEATURES = "### Added\n\n- **A feature.** Details.\n\n### Fixed\n\n- A bug. (#12)\n"


class TestLatestTag:
    def test_highest_final_version_wins(self):
        assert latest_tag(TAGS) == ("v0.5.0", (0, 5, 0), True)

    def test_a_final_release_outranks_its_pre_release(self):
        assert latest_tag(["v0.6.0rc1", "v0.6.0"]) == ("v0.6.0", (0, 6, 0), True)
        assert latest_tag(["v0.5.0", "v0.6.0rc1"]) == ("v0.6.0rc1", (0, 6, 0), False)

    def test_no_tags_is_version_zero(self):
        assert latest_tag([]) == (None, (0, 0, 0), True)
        assert latest_tag(["latest", "release-2026"]) == (None, (0, 0, 0), True)


class TestParseTitle:
    @pytest.mark.parametrize("title", [
        "release: v0.6.0",
        "release: v0.6.0 (#95)",
        "Merge pull request #42 from ssf0409/release/v0.6.0",
    ])
    def test_release_commits(self, title):
        assert parse_title(title + "\n\nbody").kind == "release_commit"

    @pytest.mark.parametrize("title", [
        "fix: x [release: skip]",
        "fix: x [Release: SKIP]",
        "[skip release] fix: x",
        "fix: x [no release]",
        "fix: x [release: none]",
    ])
    def test_skip_markers(self, title):
        assert parse_title(title).kind == "skip"

    def test_bump_and_version_markers(self):
        assert parse_title("feat: x [release: minor]") == module.Marker("bump", "minor")
        assert parse_title("feat: x [release: MAJOR]") == module.Marker("bump", "major")
        assert parse_title("feat: x [release: 1.0.0]") == module.Marker("version", "1.0.0")
        assert parse_title("feat: x [release: v1.0.0rc1]") == module.Marker("version", "1.0.0rc1")

    def test_only_the_title_is_read(self):
        assert parse_title("feat: x\n\nThe body says [release: skip] but that is prose.").kind is None
        assert parse_title("").kind is None

    def test_repeated_identical_markers_are_one_request(self):
        assert parse_title("x [release: patch] [release: patch]") == module.Marker("bump", "patch")

    def test_conflicting_or_unknown_markers_are_errors(self):
        with pytest.raises(ReleaseError, match="more than one thing"):
            parse_title("x [release: skip] [release: minor]")
        with pytest.raises(ReleaseError, match="unknown release marker"):
            parse_title("x [release: soon]")


class TestInferredBump:
    def test_features_or_changes_are_minor(self):
        assert inferred_bump(FEATURES) == ("minor", "[Unreleased] has entries under Added")
        assert inferred_bump("### Changed\n\n- x\n\n### Removed\n\n- y\n")[0] == "minor"
        assert inferred_bump("### Deprecated\n\n- x\n")[0] == "minor"

    def test_fixes_only_are_patch(self):
        assert inferred_bump(FIXES) == ("patch", "[Unreleased] has entries only under Fixed")
        assert inferred_bump("### Fixed\n\n- x\n\n### Security\n\n- y\n")[0] == "patch"

    def test_an_empty_heading_does_not_count(self):
        assert inferred_bump("### Added\n\n### Fixed\n\n- x\n")[0] == "patch"

    def test_entries_without_a_heading_cannot_be_called_fixes(self):
        assert inferred_bump("- something\n") == (
            "minor", "[Unreleased] has entries outside any ### heading"
        )


class TestDecide:
    def test_fixes_make_a_patch_release(self):
        decision = decide(changelog(FIXES), "fix: x (#12)", TAGS)
        assert decision.release is True
        assert (decision.version, decision.bump) == ("0.5.1", "patch")
        assert decision.reason == "[Unreleased] has entries only under Fixed: patch release after v0.5.0"

    def test_features_make_a_minor_release(self):
        decision = decide(changelog(FEATURES), "feat: x", TAGS)
        assert (decision.version, decision.bump) == ("0.6.0", "minor")

    def test_nothing_queued_is_no_release(self):
        decision = decide(changelog("### Added\n"), "feat: x", TAGS)
        assert decision.release is False and "no entries" in decision.reason
        assert decision.lines() == ["release=false", "reason=" + decision.reason]

    def test_release_commits_and_skip_markers_never_release(self):
        assert decide(changelog(FEATURES), "release: v0.6.0 (#99)", TAGS).release is False
        assert "Release tag workflow" in decide(changelog(FEATURES), "release: v0.6.0", TAGS).reason
        assert decide(changelog(FEATURES), "feat: x [release: skip]", TAGS).release is False

    def test_bump_markers_override_the_headings(self):
        assert decide(changelog(FIXES), "fix: x [release: minor]", TAGS).version == "0.6.0"
        assert decide(changelog(FIXES), "fix: x [release: major]", TAGS).version == "1.0.0"
        assert decide(changelog(FEATURES), "feat: x [release: patch]", TAGS).version == "0.5.1"

    def test_a_named_version_is_used_as_is(self):
        decision = decide(changelog(FIXES), "fix: x [release: 2.0.0rc1]", TAGS)
        assert (decision.version, decision.bump) == ("2.0.0rc1", "explicit")
        assert decision.lines()[:3] == ["release=true", "version=2.0.0rc1", "bump=explicit"]

    def test_a_named_version_must_be_new_and_not_older(self):
        with pytest.raises(ReleaseError, match="older than the latest tag v0.5.0"):
            decide(changelog(FIXES), "x [release: 0.4.1]", TAGS)
        with pytest.raises(ReleaseError, match="tag v0.5.0 already exists"):
            decide(changelog(FIXES), "x [release: 0.5.0]", TAGS)
        with pytest.raises(ReleaseError, match="already has a section for 0.5.0"):
            decide(changelog(FIXES), "x [release: 0.5.0]", ["v0.4.0"])

    def test_a_pre_release_tag_waits_for_a_maintainer(self):
        tags = [*TAGS, "v0.6.0rc1"]
        decision = decide(changelog(FIXES), "fix: x", tags)
        assert decision.release is False and "v0.6.0rc1 is a pre-release" in decision.reason
        # Naming the version still works: that is how the rc becomes final.
        assert decide(changelog(FIXES), "fix: x [release: 0.6.0]", tags).version == "0.6.0"

    def test_first_release_ever(self):
        assert decide(changelog(FEATURES), "feat: x", []).version == "0.1.0"
        assert decide(changelog(FIXES), "fix: x", []).version == "0.0.1"
        assert decide(changelog(FIXES), "fix: x", []).reason.endswith("as the first release tag")

    def test_no_unreleased_section_is_an_error(self):
        with pytest.raises(ReleaseError, match="no '## \\[Unreleased\\]'"):
            decide("# Changelog\n\n## [0.5.0] - 2026-09-06\n\n- Older.\n", "x", TAGS)

    def test_reason_is_one_line_for_github_output(self):
        decision = module.Decision(True, "two\nlines  here", "0.1.0", "minor")
        assert decision.lines() == [
            "release=true", "version=0.1.0", "bump=minor", "reason=two lines here"
        ]


class TestCommandLine:
    def _run(self, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, timeout=60,
        )

    def test_prints_github_output_lines(self, tmp_path: Path):
        (tmp_path / "CHANGELOG.md").write_text(changelog(FEATURES))
        (tmp_path / "msg.txt").write_text("feat: x (#1)\n\nBody mentions [release: skip].\n")
        result = self._run("--message-file", "msg.txt", "--tags", *TAGS, cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [
            "release=true",
            "version=0.6.0",
            "bump=minor",
            "reason=[Unreleased] has entries under Added: minor release after v0.5.0",
        ]
        skipped = self._run("--message", "feat: x [release: skip]", "--tags", cwd=tmp_path)
        assert skipped.returncode == 0 and skipped.stdout.startswith("release=false\n")

    def test_errors_exit_1_or_2(self, tmp_path: Path):
        (tmp_path / "CHANGELOG.md").write_text(changelog(FIXES))
        bad = self._run("--message", "x [release: 0.4.1]", "--tags", *TAGS, cwd=tmp_path)
        assert bad.returncode == 1 and "older than the latest tag" in bad.stderr
        assert self._run("--message", "x", "--changelog", "missing.md", cwd=tmp_path).returncode == 2
        assert self._run("--message-file", "missing.txt", cwd=tmp_path).returncode == 2
        assert self._run("--tags", cwd=tmp_path).returncode == 2  # a message is required

    def test_real_repository_state_is_decidable(self):
        # Tags come from git; the changelog is the real one. Whatever state
        # the repository is in, the script must decide without an error:
        # a release (above the latest tag) or "nothing to release".
        result = self._run("--message", "chore: check the release state", cwd=REPO)
        assert result.returncode == 0, result.stderr
        output = dict(line.split("=", 1) for line in result.stdout.splitlines())
        tags = subprocess.run(
            ["git", "tag", "--list", "v*"], cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout.split()
        _tag, base, _final = latest_tag(tags)
        if output["release"] == "true":
            assert module.version_key(output["version"]) > base
        else:
            assert "no entries" in output["reason"] or "pre-release" in output["reason"]


class TestWorkflowWiring:
    """The workflow, the CI workflow it waits for, and the scripts agree."""

    @staticmethod
    def _load(name: str) -> dict:
        data = yaml.safe_load((REPO / ".github" / "workflows" / name).read_text(encoding="utf-8"))
        data["on"] = data.pop(True, None) or data.get("on")  # YAML reads `on:` as True
        return data

    def test_release_auto_waits_for_the_ci_workflow_by_name(self):
        auto = self._load("release-auto.yml")
        assert auto["on"]["workflow_run"]["workflows"] == [self._load("ci.yml")["name"]]
        assert auto["on"]["workflow_run"]["branches"] == ["main"]
        assert auto["on"]["workflow_run"]["types"] == ["completed"]
        job = auto["jobs"]["release"]
        assert "conclusion == 'success'" in job["if"] and "event == 'push'" in job["if"]
        assert job["permissions"] == {"contents": "write", "actions": "write"}

    def test_release_auto_uses_the_scripts_and_the_other_workflows(self):
        steps = self._load("release-auto.yml")["jobs"]["release"]["steps"]
        script = "\n".join(step.get("run", "") for step in steps)
        for needle in (
            "scripts/next_version.py --message-file",
            "scripts/prepare_release.py --version",
            "scripts/release_notes.py --version",
            "gh workflow run release.yml",
            "-f publish=true",
            "gh workflow run release-prepare.yml",
        ):
            assert needle in script, needle
        # The dispatch targets exist with the inputs the script passes.
        release = self._load("release.yml")
        assert "publish" in release["on"]["workflow_dispatch"]["inputs"]
        prepare = self._load("release-prepare.yml")
        assert "version" in prepare["on"]["workflow_dispatch"]["inputs"]

    def test_release_commit_title_matches_the_release_tag_workflow(self):
        # The commit release-auto pushes must be one release-tag.yml would
        # also recognise, so a manual re-run of either finds the same tag.
        tag = self._load("release-tag.yml")["jobs"]["tag"]
        assert "startsWith(github.event.head_commit.message, 'release: v')" in tag["if"]
        auto = "\n".join(
            step.get("run", "") for step in self._load("release-auto.yml")["jobs"]["release"]["steps"]
        )
        assert 'git commit -m "release: v$VERSION"' in auto
        assert 'git tag -a "v$VERSION" -m "release: v$VERSION"' in auto
