"""Tests for scripts/check_changelog.py, which CI runs on every pull request."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_changelog.py"
REPO = SCRIPT.parents[1]
spec = importlib.util.spec_from_file_location("check_changelog", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
check = module.check

BASE = """\
# Changelog

## [Unreleased]

### Fixed

- A bug. (#12)

## [0.5.0] - 2026-09-06

### Added

- **A feature.** Details.
"""

RELEASED = """\
# Changelog

## [Unreleased]

## [0.6.0] - 2026-09-15

### Fixed

- A bug. (#12)

## [0.5.0] - 2026-09-06

### Added

- **A feature.** Details.
"""


class TestCheck:
    def test_unchanged_and_new_unreleased_entries_are_fine(self):
        assert check(BASE, BASE) == []
        head = BASE.replace("- A bug. (#12)\n", "- A bug. (#12)\n- Another bug. (#13)\n")
        assert check(BASE, head) == []

    def test_an_entry_added_to_a_released_section_is_refused(self):
        head = BASE.replace("- **A feature.** Details.\n", "- **A feature.** Details.\n- Late entry. (#14)\n")
        problems = check(BASE, head)
        assert len(problems) == 1
        assert problems[0].startswith("released section [0.5.0] gained 1 entry (- Late entry. (#14))")
        assert "move new entries under [Unreleased]" in problems[0]

    def test_wording_fixes_in_a_released_section_are_fine(self):
        head = BASE.replace("- **A feature.** Details.", "- **A feature.** Better details.")
        assert check(BASE, head) == []

    def test_a_release_commit_moves_everything(self):
        assert check(BASE, RELEASED) == []

    def test_entries_left_behind_after_a_release_are_refused(self):
        # A pull request merged after the release: its entry was added to the
        # base's [Unreleased], and a git merge left it inside [0.6.0].
        head = RELEASED.replace("- A bug. (#12)\n", "- A bug. (#12)\n- Mine. (#15)\n")
        problems = check(RELEASED, head)
        assert len(problems) == 1 and "released section [0.6.0] gained 1 entry" in problems[0]

    def test_a_new_dated_section_with_entries_still_queued_is_refused(self):
        head = BASE.replace(
            "## [0.5.0]", "## [0.5.1] - 2026-09-15\n\n### Fixed\n\n- Hand-made. (#16)\n\n## [0.5.0]"
        )
        problems = check(BASE, head)
        assert len(problems) == 1 and "section [0.5.1] is new but [Unreleased] still has 1 entry" in problems[0]

    def test_a_missing_unreleased_section_is_refused(self):
        head = BASE.replace("## [Unreleased]\n\n### Fixed\n\n- A bug. (#12)\n\n", "")
        assert any("no '## [Unreleased]' section" in p for p in check(BASE, head))

    def test_long_entries_are_shortened_in_the_message(self):
        entry = "- " + "x" * 100 + "\n"
        head = BASE.replace("- **A feature.** Details.\n", "- **A feature.** Details.\n" + entry)
        [problem] = check(BASE, head)
        assert "x" * 68 + "..." in problem and "x" * 80 not in problem


class TestCommandLine:
    def _run(self, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, timeout=60,
        )

    def test_files_and_exit_codes(self, tmp_path: Path):
        (tmp_path / "base.md").write_text(BASE)
        (tmp_path / "CHANGELOG.md").write_text(BASE.replace("- A bug. (#12)\n", "- A bug. (#12)\n- New. (#1)\n"))
        ok = self._run("--base-file", "base.md", cwd=tmp_path)
        assert ok.returncode == 0 and ok.stdout.startswith("ok: 2 entries under [Unreleased]")
        (tmp_path / "CHANGELOG.md").write_text(
            BASE.replace("- **A feature.** Details.\n", "- **A feature.** Details.\n- Late. (#2)\n")
        )
        bad = self._run("--base-file", "base.md", cwd=tmp_path)
        assert bad.returncode == 1 and "error: released section [0.5.0] gained 1 entry" in bad.stderr
        assert self._run("--base-file", "missing.md", cwd=tmp_path).returncode == 2
        assert self._run("--base", "origin/main", "--head-file", "nope.md", cwd=tmp_path).returncode == 2

    def test_base_from_a_git_ref(self, tmp_path: Path):
        env_git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
        (tmp_path / "CHANGELOG.md").write_text(BASE)
        subprocess.run(["git", "add", "CHANGELOG.md"], cwd=tmp_path, check=True)
        subprocess.run([*env_git, "commit", "-q", "-m", "base"], cwd=tmp_path, check=True)
        (tmp_path / "CHANGELOG.md").write_text(
            BASE.replace("- **A feature.** Details.\n", "- **A feature.** Details.\n- Late. (#2)\n")
        )
        result = self._run("--base", "main", cwd=tmp_path)
        assert result.returncode == 1 and "[0.5.0] gained 1 entry" in result.stderr
        assert self._run("--base", "no-such-ref", cwd=tmp_path).returncode == 2

    def test_real_changelog_against_itself(self):
        result = self._run("--base-file", "CHANGELOG.md", cwd=REPO)
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith("ok: ")


def test_ci_runs_the_check_on_pull_requests():
    import yaml

    ci = yaml.safe_load((REPO / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = ci["jobs"]["lint"]["steps"]
    step = next(s for s in steps if "check_changelog.py" in s.get("run", ""))
    assert step["if"] == "github.event_name == 'pull_request'"
    assert "--base" in step["run"] and "git fetch" in step["run"]
