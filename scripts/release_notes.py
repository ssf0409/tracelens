#!/usr/bin/env python3
"""Extract one version's notes from CHANGELOG.md for the GitHub Release.

Used by ``.github/workflows/release.yml`` so the GitHub Release tells the
same story as the changelog. The script refuses (exit 1) when the version has
no dated ``## [X.Y.Z] - YYYY-MM-DD`` section instead of inventing notes;
``--allow-unreleased`` exists for dry runs of the workflow on a branch, where
the built version is a development version and the ``[Unreleased]`` section
stands in for it, clearly labelled.

Usage:
    python scripts/release_notes.py --version 0.5.0 --output notes.md
    python scripts/release_notes.py --version 0.5.0rc1 --print prerelease

Exit codes: 0 notes written or printed; 1 no usable section; 2 usage error.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^## \[(?P<name>[^\]]+)\](?: - (?P<date>.+))?\s*$")
FINAL_VERSION = re.compile(r"^\d+\.\d+\.\d+$")
REPO_URL = "https://github.com/ssf0409/tracelens"


def sections(changelog: str) -> dict[str, str]:
    """Map every ``## [name]`` header to its body text, in file order."""
    result: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for line in changelog.splitlines():
        match = HEADER.match(line)
        if match:
            if current is not None:
                result[current] = "\n".join(lines).strip() + "\n"
            current = match.group("name")
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        result[current] = "\n".join(lines).strip() + "\n"
    return result


def is_prerelease(version: str) -> bool:
    """Anything but a final ``X.Y.Z`` (alpha, beta, rc, dev, post) is a pre-release."""
    return FINAL_VERSION.match(version) is None


def release_notes(changelog: str, version: str, *, allow_unreleased: bool = False) -> str:
    """The release body for ``version``: its changelog section plus install links.

    Raises:
        LookupError: The version has no section (or it is empty) and no
            fallback applies. The message names the sections that exist.
    """
    found = sections(changelog)
    body = found.get(version)
    prefix = ""
    if body is None and allow_unreleased and "Unreleased" in found:
        body = found["Unreleased"]
        prefix = (
            f"> Dry run: CHANGELOG.md has no section for {version}; "
            "this is the [Unreleased] section.\n\n"
        )
    elif body is None:
        available = ", ".join(name for name in found if name != "Unreleased") or "none"
        raise LookupError(
            f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' section (found: {available}); "
            "add the dated section before tagging"
        )
    if not body.strip():
        raise LookupError(f"the changelog section for {version} is empty")
    trailer = f"[Full changelog]({REPO_URL}/blob/main/CHANGELOG.md)"
    if not prefix:
        trailer = (
            f"Install: `pip install tracelens=={version}` · "
            f"[PyPI](https://pypi.org/project/tracelens/{version}/) · " + trailer
        )
    return prefix + body.rstrip() + "\n\n" + trailer + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", required=True, help="Package version, e.g. 0.5.0")
    parser.add_argument("--changelog", default="CHANGELOG.md", help="Path to CHANGELOG.md")
    parser.add_argument("--output", default=None, help="Write the notes here instead of stdout")
    parser.add_argument(
        "--allow-unreleased", action="store_true",
        help="Dry runs only: fall back to the [Unreleased] section, labelled as such",
    )
    parser.add_argument(
        "--print", choices=["notes", "prerelease"], default="notes",
        help="'prerelease' prints true/false for the version instead of notes",
    )
    args = parser.parse_args(argv)
    if args.print == "prerelease":
        print("true" if is_prerelease(args.version) else "false")
        return 0
    try:
        changelog = Path(args.changelog).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        notes = release_notes(changelog, args.version, allow_unreleased=args.allow_unreleased)
    except LookupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.output:
        Path(args.output).write_text(notes, encoding="utf-8")
        print(f"wrote {args.output} ({len(notes)} characters)")
    else:
        print(notes, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
