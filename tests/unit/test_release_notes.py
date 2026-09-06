"""Tests for scripts/release_notes.py, the changelog extractor the release workflow uses."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
assert spec is not None and spec.loader is not None
release_notes_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_notes_module)
is_prerelease = release_notes_module.is_prerelease
release_notes = release_notes_module.release_notes
sections = release_notes_module.sections

CHANGELOG = """\
# Changelog

Intro text.

## [Unreleased]

### Added

- Something in progress.

## [0.5.0] - 2026-09-20

### Added

- **A feature.** With `code` and a [link](https://example.com).

### Fixed

- A bug. (#12)

## [0.4.0] - 2026-07-19

- Older.
"""


class TestSections:
    def test_maps_headers_to_bodies_in_order(self):
        found = sections(CHANGELOG)
        assert list(found) == ["Unreleased", "0.5.0", "0.4.0"]
        assert found["0.4.0"] == "- Older.\n"
        assert found["0.5.0"].startswith("### Added\n\n- **A feature.**")
        assert found["0.5.0"].endswith("- A bug. (#12)\n")

    def test_ignores_text_before_the_first_header_and_other_headings(self):
        assert sections("# Title\n\ntext\n\n### not a version\n") == {}


class TestPrerelease:
    @pytest.mark.parametrize("version", ["0.5.0", "1.0.0", "10.20.30"])
    def test_final_versions(self, version):
        assert not is_prerelease(version)

    @pytest.mark.parametrize("version", ["0.5.0rc1", "0.5.0a1", "0.5.0b2", "0.5.0.dev3", "0.5.0.post1"])
    def test_everything_else(self, version):
        assert is_prerelease(version)


class TestReleaseNotes:
    def test_section_plus_install_links(self):
        notes = release_notes(CHANGELOG, "0.5.0")
        assert notes.startswith("### Added\n\n- **A feature.**")
        assert "- A bug. (#12)\n\n" in notes
        assert "Install: `pip install tracelens==0.5.0`" in notes
        assert "https://pypi.org/project/tracelens/0.5.0/" in notes
        assert notes.endswith("CHANGELOG.md)\n")
        assert "Older" not in notes and "in progress" not in notes

    def test_missing_section_is_an_error_naming_what_exists(self):
        with pytest.raises(LookupError, match=r"no '## \[0\.6\.0\] - YYYY-MM-DD' section \(found: 0\.5\.0, 0\.4\.0\)"):
            release_notes(CHANGELOG, "0.6.0")

    def test_empty_section_is_an_error(self):
        with pytest.raises(LookupError, match="section for 0.7.0 is empty"):
            release_notes("## [0.7.0] - 2026-01-01\n\n\n## [0.6.0] - 2025-12-01\n\n- x\n", "0.7.0")

    def test_dry_run_falls_back_to_unreleased_and_says_so(self):
        notes = release_notes(CHANGELOG, "0.5.1.dev4", allow_unreleased=True)
        assert notes.startswith("> Dry run: CHANGELOG.md has no section for 0.5.1.dev4")
        assert "- Something in progress." in notes
        assert "Install:" not in notes  # a dev version is not on PyPI
        # a real section still wins over the fallback
        assert "in progress" not in release_notes(CHANGELOG, "0.5.0", allow_unreleased=True)

    def test_fallback_needs_an_unreleased_section(self):
        with pytest.raises(LookupError):
            release_notes("## [0.4.0] - 2026-07-19\n\n- Older.\n", "0.5.0", allow_unreleased=True)


def test_dry_run_with_empty_unreleased_renders_a_placeholder():
    empty = CHANGELOG.replace("### Added\n\n- Something in progress.\n\n## [0.5.0]", "## [0.5.0]")
    assert empty != CHANGELOG and not sections(empty)["Unreleased"].strip()
    notes = release_notes(empty, "0.6.0.dev1", allow_unreleased=True)
    assert notes.startswith(
        "> Dry run: CHANGELOG.md has no section for 0.6.0.dev1; "
        "the [Unreleased] section is empty, so there is nothing to release yet."
    )
    assert "[Full changelog]" in notes and "pip install" not in notes
    with pytest.raises(LookupError, match="add the dated section"):
        release_notes(empty, "0.6.0.dev1")  # no fallback without the flag


class TestCommandLine:
    def _run(self, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args], cwd=cwd, capture_output=True, text=True, timeout=60,
        )

    def test_writes_notes_file_and_prints_prerelease(self, tmp_path: Path):
        (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
        result = self._run("--version", "0.5.0", "--output", "notes.md", cwd=tmp_path)
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "notes.md").read_text() == release_notes(CHANGELOG, "0.5.0")
        assert self._run("--version", "0.5.0rc1", "--print", "prerelease", cwd=tmp_path).stdout == "true\n"
        assert self._run("--version", "0.5.0", "--print", "prerelease", cwd=tmp_path).stdout == "false\n"

    def test_missing_section_exits_1_and_missing_file_exits_2(self, tmp_path: Path):
        (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
        missing = self._run("--version", "9.9.9", cwd=tmp_path)
        assert missing.returncode == 1 and "add the dated section before tagging" in missing.stderr
        no_file = self._run("--version", "0.5.0", "--changelog", "nope.md", cwd=tmp_path)
        assert no_file.returncode == 2 and "error:" in no_file.stderr

    def test_real_changelog_renders_its_latest_release_and_a_dry_run(self):
        # Must hold in every release state: with entries under [Unreleased]
        # and right after a release, when that section is empty.
        repo = SCRIPT.parents[1]
        text = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
        latest = next(name for name in sections(text) if name != "Unreleased")
        released = self._run("--version", latest, cwd=repo)
        assert released.returncode == 0 and f"tracelens=={latest}" in released.stdout
        dry = self._run("--version", "0.0.0.dev0", "--allow-unreleased", cwd=repo)
        assert dry.returncode == 0 and dry.stdout.startswith("> Dry run:"), dry.stderr
