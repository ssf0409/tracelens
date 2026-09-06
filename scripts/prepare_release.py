#!/usr/bin/env python3
"""Turn the changelog's ``[Unreleased]`` section into a dated release section.

Used by ``.github/workflows/release-prepare.yml`` to open the release pull
request; also usable by hand. It moves everything under ``## [Unreleased]``
into ``## [X.Y.Z] - YYYY-MM-DD`` (leaving an empty Unreleased section
behind), optionally prefixed with a one-paragraph summary, and refuses to
release nothing: an empty Unreleased section, a version that already has a
section, or a malformed version is an error before any file is written.

Usage:
    python scripts/prepare_release.py --version 0.5.0 [--summary "..."]
    python scripts/prepare_release.py --version 0.5.0 --check   # validate only

Exit codes: 0 done (or valid); 1 nothing to release / bad version; 2 usage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

UNRELEASED = "## [Unreleased]"
VERSION = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+|\.post\d+)?$")
SECTION = re.compile(r"^## \[(?P<name>[^\]]+)\]")


class ReleaseError(ValueError):
    """The changelog cannot be released as it stands; the message says why."""


def parse_version(version: str) -> tuple[int, int, int]:
    """The (major, minor, patch) of a PEP 440-style ``X.Y.Z[suffix]`` string."""
    if VERSION.match(version) is None:
        raise ReleaseError(
            f"version {version!r} is not X.Y.Z with an optional a/b/rc/.dev/.post suffix"
        )
    major, minor, patch = version.split(".")[:3]
    return int(major), int(minor), int(re.match(r"\d+", patch).group())  # type: ignore[union-attr]


def split_unreleased(changelog: str) -> tuple[str, str, str]:
    """``(head, unreleased_body, tail)`` around the Unreleased section."""
    lines = changelog.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.rstrip() == UNRELEASED), None)
    if start is None:
        raise ReleaseError("CHANGELOG.md has no '## [Unreleased]' section")
    end = next(
        (i for i in range(start + 1, len(lines)) if SECTION.match(lines[i])), len(lines)
    )
    return "".join(lines[: start + 1]), "".join(lines[start + 1 : end]), "".join(lines[end:])


def prepare(
    changelog: str,
    version: str,
    *,
    date: dt.date | None = None,
    summary: str | None = None,
) -> tuple[str, str]:
    """``(new_changelog, released_section_body)`` for ``version``.

    Raises:
        ReleaseError: malformed version, a section for it already exists,
            or the Unreleased section has no entries.
    """
    parse_version(version)
    if re.search(rf"^## \[{re.escape(version)}\]", changelog, flags=re.MULTILINE):
        raise ReleaseError(f"CHANGELOG.md already has a section for {version}")
    head, body, tail = split_unreleased(changelog)
    if not any(line.startswith("- ") for line in body.splitlines()):
        raise ReleaseError("the [Unreleased] section has no entries; nothing to release")
    when = (date or dt.datetime.now(dt.UTC).date()).isoformat()
    section_body = body.strip("\n") + "\n"
    if summary:
        section_body = summary.strip() + "\n\n" + section_body
    new_section = f"## [{version}] - {when}\n\n{section_body}\n"
    return head + "\n" + new_section + tail, section_body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", required=True, help="Version to release, e.g. 0.5.0")
    parser.add_argument("--changelog", default="CHANGELOG.md")
    parser.add_argument("--summary", default=None, help="One paragraph placed above the entries")
    parser.add_argument("--date", default=None, help="YYYY-MM-DD (default: today, UTC)")
    parser.add_argument("--check", action="store_true", help="Validate without writing")
    args = parser.parse_args(argv)
    path = Path(args.changelog)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        date = dt.date.fromisoformat(args.date) if args.date else None
    except ValueError as exc:
        print(f"error: --date: {exc}", file=sys.stderr)
        return 2
    try:
        updated, section = prepare(text, args.version, date=date, summary=args.summary)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print(f"ok: {args.version} would release {section.count(chr(10))} line(s) of notes")
        return 0
    path.write_text(updated, encoding="utf-8")
    entries = sum(1 for line in section.splitlines() if line.startswith("- "))
    print(f"released {args.version} in {path}: {entries} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
