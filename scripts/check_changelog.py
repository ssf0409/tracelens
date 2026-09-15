#!/usr/bin/env python3
"""Check that changelog entries land under ``[Unreleased]``, never in a release.

Used by CI on every pull request (``ci.yml``, lint job) and usable locally:

    python scripts/check_changelog.py --base origin/main
    python scripts/check_changelog.py --base-file old.md --head-file CHANGELOG.md

Compares the pull request's ``CHANGELOG.md`` with the base branch's:

- A released section (``## [X.Y.Z] - date``) present on both sides may not
  gain entries (top-level ``- `` bullets). Wording fixes are fine. New
  bullets are not: after a release, a merge or rebase can leave a pull
  request's entries inside the section that was just released, where they
  would never ship and would misdescribe a published version.
- A released section that is new on the pull request side is a release
  commit (``release: vX.Y.Z``), allowed only when ``[Unreleased]`` is left
  empty: a release moves every queued entry, and new entries go under
  ``[Unreleased]``.
- ``[Unreleased]`` must exist.

Exit codes: 0 placement is fine; 1 misplaced entries; 2 usage/IO error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_notes import sections  # noqa: E402

UNRELEASED = "Unreleased"


def bullets(body: str) -> list[str]:
    """The top-level entries of a section body: lines starting with ``- ``."""
    return [line.rstrip() for line in body.splitlines() if line.startswith("- ")]


def check(base: str, head: str) -> list[str]:
    """Problems with the entries ``head`` adds relative to ``base``.

    An empty list means every new entry sits under ``[Unreleased]``.
    """
    base_sections = sections(base)
    head_sections = sections(head)
    problems: list[str] = []
    if UNRELEASED not in head_sections:
        problems.append("CHANGELOG.md has no '## [Unreleased]' section; new entries go there")
    queued = bullets(head_sections.get(UNRELEASED, ""))
    for name, body in head_sections.items():
        if name == UNRELEASED:
            continue
        entries = bullets(body)
        if name in base_sections:
            before = bullets(base_sections[name])
            if len(entries) > len(before):
                added = [entry for entry in entries if entry not in before] or entries[len(before):]
                shown = "; ".join(f"{entry[:70]}..." if len(entry) > 70 else entry for entry in added)
                problems.append(
                    f"released section [{name}] gained {len(entries) - len(before)} entr"
                    f"{'y' if len(entries) - len(before) == 1 else 'ies'} ({shown}): "
                    "move new entries under [Unreleased]; a released section is frozen"
                )
        elif queued:
            problems.append(
                f"section [{name}] is new but [Unreleased] still has {len(queued)} entr"
                f"{'y' if len(queued) == 1 else 'ies'}: a release moves every queued entry "
                "into the new section; anything else goes under [Unreleased]"
            )
    return problems


def read_git(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True
    )
    return result.stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    base = parser.add_mutually_exclusive_group(required=True)
    base.add_argument("--base", help="Git ref of the base branch, e.g. origin/main")
    base.add_argument("--base-file", help="The base branch's changelog as a file")
    parser.add_argument("--head-file", default="CHANGELOG.md", help="The changelog to check")
    parser.add_argument(
        "--path", default="CHANGELOG.md", help="Path of the changelog inside the repository"
    )
    args = parser.parse_args(argv)
    try:
        head_text = Path(args.head_file).read_text(encoding="utf-8")
        base_text = (
            read_git(args.base, args.path)
            if args.base is not None
            else Path(args.base_file).read_text(encoding="utf-8")
        )
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"error: git show {args.base}:{args.path}: {exc.stderr.strip()}", file=sys.stderr)
        return 2
    problems = check(base_text, head_text)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    queued = len(bullets(sections(head_text).get(UNRELEASED, "")))
    print(f"ok: {queued} entr{'y' if queued == 1 else 'ies'} under [Unreleased]; released sections unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
