"""Tests for scripts/prepare_release.py, which the release-prepare workflow runs."""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_release.py"
spec = importlib.util.spec_from_file_location("prepare_release", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ReleaseError = module.ReleaseError
prepare = module.prepare
parse_version = module.parse_version
split_unreleased = module.split_unreleased

CHANGELOG = """\
# Changelog

Intro.

## [Unreleased]

### Added

- **A feature.** Details.

### Fixed

- A bug. (#12)

## [0.4.0] - 2026-07-19

- Older.
"""


class TestParseVersion:
    @pytest.mark.parametrize("version", ["0.5.0", "1.0.0rc1", "0.5.0.dev3", "2.0.0.post1"])
    def test_accepts_pep440_shapes(self, version):
        assert parse_version(version)[:2] == tuple(int(x) for x in version.split(".")[:2])

    @pytest.mark.parametrize("version", ["v0.5.0", "0.5", "0.5.0-beta", "latest", ""])
    def test_rejects_other_shapes(self, version):
        with pytest.raises(ReleaseError, match="not X.Y.Z"):
            parse_version(version)


class TestPrepare:
    def test_moves_unreleased_into_a_dated_section(self):
        updated, section = prepare(CHANGELOG, "0.5.0", date=dt.date(2026, 9, 6))
        assert "## [Unreleased]\n\n## [0.5.0] - 2026-09-06\n\n### Added\n" in updated
        assert section.startswith("### Added\n") and section.endswith("- A bug. (#12)\n")
        assert updated.count("- **A feature.**") == 1
        assert updated.endswith("## [0.4.0] - 2026-07-19\n\n- Older.\n")
        # Unreleased is now empty, ready for the next cycle.
        head, body, _ = module.split_unreleased(updated)
        assert body.strip() == ""

    def test_summary_goes_above_the_entries(self):
        updated, section = prepare(CHANGELOG, "0.5.0", date=dt.date(2026, 9, 6), summary="Big release.")
        assert section.startswith("Big release.\n\n### Added\n")
        assert "## [0.5.0] - 2026-09-06\n\nBig release.\n\n### Added" in updated

    def test_refuses_to_release_nothing(self):
        empty = "# Changelog\n\n## [Unreleased]\n\n### Added\n\n## [0.4.0] - 2026-07-19\n\n- Older.\n"
        with pytest.raises(ReleaseError, match="nothing to release"):
            prepare(empty, "0.5.0")

    def test_refuses_an_existing_version_and_a_missing_unreleased_section(self):
        with pytest.raises(ReleaseError, match="already has a section for 0.4.0"):
            prepare(CHANGELOG, "0.4.0")
        with pytest.raises(ReleaseError, match="no '## \\[Unreleased\\]'"):
            prepare("# Changelog\n\n## [0.4.0] - 2026-07-19\n\n- Older.\n", "0.5.0")

    def test_result_is_what_release_notes_will_extract(self):
        notes_spec = importlib.util.spec_from_file_location(
            "release_notes", SCRIPT.parent / "release_notes.py"
        )
        assert notes_spec is not None and notes_spec.loader is not None
        notes = importlib.util.module_from_spec(notes_spec)
        notes_spec.loader.exec_module(notes)
        updated, section = prepare(CHANGELOG, "0.5.0", date=dt.date(2026, 9, 6), summary="Big release.")
        assert notes.sections(updated)["0.5.0"] == section
        assert notes.release_notes(updated, "0.5.0").startswith(section.rstrip())


class TestCommandLine:
    def _run(self, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, timeout=60,
        )

    def test_check_then_write(self, tmp_path: Path):
        (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
        check = self._run("--version", "0.5.0", "--check", cwd=tmp_path)
        assert check.returncode == 0 and check.stdout.startswith("ok: 0.5.0 would release")
        assert (tmp_path / "CHANGELOG.md").read_text() == CHANGELOG  # untouched
        write = self._run("--version", "0.5.0", "--date", "2026-09-06", cwd=tmp_path)
        assert write.returncode == 0 and "released 0.5.0" in write.stdout and "2 entries" in write.stdout
        assert "## [0.5.0] - 2026-09-06" in (tmp_path / "CHANGELOG.md").read_text()

    def test_errors_exit_1_or_2(self, tmp_path: Path):
        (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
        assert self._run("--version", "0.4.0", cwd=tmp_path).returncode == 1
        assert self._run("--version", "nope", cwd=tmp_path).returncode == 1
        assert self._run("--version", "0.5.0", "--date", "yesterday", cwd=tmp_path).returncode == 2
        assert self._run("--version", "0.5.0", "--changelog", "missing.md", cwd=tmp_path).returncode == 2

    def test_real_changelog_is_well_formed_in_every_release_state(self):
        # With entries under [Unreleased] a check succeeds; right after a
        # release the section is empty and the only acceptable refusal is
        # "nothing to release". Anything else means the changelog is broken.
        repo = SCRIPT.parents[1]
        _, unreleased, tail = split_unreleased((repo / "CHANGELOG.md").read_text(encoding="utf-8"))
        assert tail.startswith("## ["), "a dated section must follow [Unreleased]"
        result = self._run("--version", "999.0.0", "--check", cwd=repo)
        if unreleased.strip():
            assert result.returncode == 0, result.stderr
            assert result.stdout.startswith("ok: 999.0.0 would release")
        else:
            assert result.returncode == 1 and "nothing to release" in result.stderr
