#!/usr/bin/env python3
"""Decide whether a commit on ``main`` releases, and which version.

Used by ``.github/workflows/release-auto.yml`` after CI passes on a push to
``main``. The decision comes from the commit title and the changelog:

- A release commit (``release: vX.Y.Z``, or the merge of a ``release/vX.Y.Z``
  pull request) never releases again: the "Release tag" workflow owns it.
- ``[release: skip]`` (also ``[skip release]``, ``[no release]``) in the
  title leaves the changes queued under ``[Unreleased]``.
- An empty ``[Unreleased]`` section is nothing to release.
- ``[release: X.Y.Z]`` names the version outright.
- ``[release: major|minor|patch]`` names the bump from the latest ``vX.Y.Z``
  tag. Without a marker the bump comes from the ``[Unreleased]`` headings:
  entries under Added, Changed, Removed, or Deprecated make a minor release,
  entries only under Fixed (or Security, and so on) a patch release.
- When the latest tag is a pre-release (``v1.0.0rc1``) nothing is released
  automatically: finish it with the "Release prepare" workflow.

Only the first line of the message (the title) is read for markers, so a
pull request description may talk about them freely.

Usage:
    python scripts/next_version.py --message-file msg.txt [--tags v0.5.0 ...]
    python scripts/next_version.py --message "fix: x [release: patch]"

Prints ``key=value`` lines for ``$GITHUB_OUTPUT``: ``release`` (true/false),
``version`` and ``bump`` when releasing, and ``reason`` always.

Exit codes: 0 decided (release or skip); 1 the title asks for something the
tags or the changelog cannot honour; 2 usage/IO error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_release import VERSION, ReleaseError, split_unreleased  # noqa: E402

TAG = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.dev\d+|\.post\d+)?)$")
FINAL = re.compile(r"^\d+\.\d+\.\d+(?:\.post\d+)?$")
RELEASE_TITLE = re.compile(r"^release: v\S+")
RELEASE_MERGE = re.compile(r"^Merge pull request #\d+ from \S+/release/v\S+")
MARKER = re.compile(
    r"\[(?:release:\s*(?P<value>[^\]\s]+)|(?P<skip>skip release|no release|release skip))\]",
    re.IGNORECASE,
)
BUMPS = ("major", "minor", "patch")
SKIP_WORDS = frozenset({"skip", "none", "no"})
# Keep a Changelog headings whose entries call for a minor release; every
# other heading (Fixed, Security, ...) is a patch release on its own.
MINOR_SECTIONS = frozenset({"added", "changed", "removed", "deprecated"})


@dataclass(frozen=True)
class Marker:
    """What the commit title asks for: ``kind`` is ``release_commit``,
    ``skip``, ``version``, ``bump``, or ``None`` for no request."""

    kind: str | None
    value: str | None = None


@dataclass(frozen=True)
class Decision:
    release: bool
    reason: str
    version: str | None = None
    bump: str | None = None

    def lines(self) -> list[str]:
        out = [f"release={'true' if self.release else 'false'}"]
        if self.version is not None:
            out.append(f"version={self.version}")
        if self.bump is not None:
            out.append(f"bump={self.bump}")
        out.append("reason=" + " ".join(self.reason.split()))
        return out


def version_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = version.split(".")[:3]
    digits = re.match(r"\d+", patch)
    assert digits is not None
    return int(major), int(minor), int(digits.group())


def latest_tag(tags: Sequence[str]) -> tuple[str | None, tuple[int, int, int], bool]:
    """``(tag, (major, minor, patch), is_final)`` of the highest ``vX.Y.Z`` tag.

    A final release outranks a pre-release of the same version; tags of any
    other shape are ignored. Without a version tag: ``(None, (0, 0, 0), True)``.
    """
    best: tuple[tuple[int, int, int], int, str] | None = None
    for tag in tags:
        match = TAG.match(tag.strip())
        if match is None:
            continue
        version = match.group("version")
        key = (version_key(version), 1 if FINAL.match(version) else 0, tag.strip())
        if best is None or key > best:
            best = key
    if best is None:
        return None, (0, 0, 0), True
    return best[2], best[0], best[1] == 1


def parse_title(message: str) -> Marker:
    """The release request in the first line of ``message``.

    Raises:
        ReleaseError: two markers ask for different things, or a marker
            names neither a bump, a skip, nor an ``X.Y.Z`` version.
    """
    title = message.strip().splitlines()[0].strip() if message.strip() else ""
    if RELEASE_TITLE.match(title) or RELEASE_MERGE.match(title):
        return Marker("release_commit", title)
    requests: list[Marker] = []
    for match in MARKER.finditer(title):
        if match.group("skip") is not None:
            requests.append(Marker("skip"))
            continue
        value = match.group("value")
        lowered = value.lower()
        if lowered in SKIP_WORDS:
            requests.append(Marker("skip"))
        elif lowered in BUMPS:
            requests.append(Marker("bump", lowered))
        elif VERSION.match(value.removeprefix("v")):
            requests.append(Marker("version", value.removeprefix("v")))
        else:
            raise ReleaseError(
                f"unknown release marker {match.group(0)!r}: use [release: skip], "
                "[release: major|minor|patch], or [release: X.Y.Z]"
            )
    distinct = sorted(set(requests), key=lambda m: (m.kind or "", m.value or ""))
    if len(distinct) > 1:
        asks = ", ".join(f"{m.kind}" + (f" {m.value}" if m.value else "") for m in distinct)
        raise ReleaseError(f"the commit title asks for more than one thing: {asks}")
    return distinct[0] if distinct else Marker(None)


def inferred_bump(unreleased_body: str) -> tuple[str, str]:
    """``(bump, why)`` from the ``[Unreleased]`` headings that have entries."""
    with_entries: list[str] = []
    flat = False
    current: str | None = None
    for line in unreleased_body.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
        elif line.startswith("- "):
            if current is None:
                flat = True
            elif current not in with_entries:
                with_entries.append(current)
    minor = [name for name in with_entries if name.lower() in MINOR_SECTIONS]
    if minor:
        return "minor", f"[Unreleased] has entries under {', '.join(minor)}"
    if flat:
        return "minor", "[Unreleased] has entries outside any ### heading"
    return "patch", f"[Unreleased] has entries only under {', '.join(with_entries)}"


def bump_version(base: tuple[int, int, int], bump: str) -> str:
    major, minor, patch = base
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def check_new(version: str, tags: Sequence[str], changelog: str, base: tuple[int, int, int]) -> None:
    """Raise ``ReleaseError`` unless ``version`` can be released after ``base``."""
    if VERSION.match(version) is None:
        raise ReleaseError(
            f"version {version!r} is not X.Y.Z with an optional a/b/rc/.dev/.post suffix"
        )
    if f"v{version}" in {tag.strip() for tag in tags}:
        raise ReleaseError(f"tag v{version} already exists")
    if version_key(version) < base:
        raise ReleaseError(f"version {version} is older than the latest tag v{'.'.join(map(str, base))}")
    if re.search(rf"^## \[{re.escape(version)}\]", changelog, flags=re.MULTILINE):
        raise ReleaseError(f"CHANGELOG.md already has a section for {version}")


def decide(changelog: str, message: str, tags: Sequence[str]) -> Decision:
    """The release decision for the commit with ``message`` at the tip of main.

    Raises:
        ReleaseError: the title asks for something impossible, or the
            changelog has no ``[Unreleased]`` section.
    """
    marker = parse_title(message)
    if marker.kind == "release_commit":
        return Decision(
            False, f"{marker.value!r} is a release commit; the Release tag workflow publishes it"
        )
    if marker.kind == "skip":
        return Decision(False, "the commit title asks to skip the release")
    _head, body, _tail = split_unreleased(changelog)
    if not any(line.startswith("- ") for line in body.splitlines()):
        return Decision(False, "the [Unreleased] section has no entries; nothing to release")
    tag, base, final = latest_tag(tags)
    since = f"after {tag}" if tag else "as the first release tag"
    if marker.kind == "version":
        assert marker.value is not None
        check_new(marker.value, tags, changelog, base)
        return Decision(
            True, f"the commit title names the version {since}", marker.value, "explicit"
        )
    if not final:
        return Decision(
            False,
            f"the latest tag {tag} is a pre-release; finish it with the Release prepare workflow",
        )
    if marker.kind == "bump":
        assert marker.value is not None
        bump, why = marker.value, f"the commit title asks for a {marker.value} release"
    else:
        bump, why = inferred_bump(body)
    version = bump_version(base, bump)
    check_new(version, tags, changelog, base)
    return Decision(True, f"{why}: {bump} release {since}", version, bump)


def git_tags() -> list[str]:
    result = subprocess.run(
        ["git", "tag", "--list", "v*"], capture_output=True, text=True, check=True
    )
    return result.stdout.split()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--changelog", default="CHANGELOG.md")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message", help="The commit message (only its first line is read)")
    source.add_argument("--message-file", help="File holding the commit message")
    parser.add_argument(
        "--tags", nargs="*", default=None,
        help="Existing tags (default: git tag --list 'v*' in the current repository)",
    )
    args = parser.parse_args(argv)
    try:
        changelog = Path(args.changelog).read_text(encoding="utf-8")
        message = (
            Path(args.message_file).read_text(encoding="utf-8")
            if args.message_file is not None
            else args.message
        )
        tags = git_tags() if args.tags is None else args.tags
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        decision = decide(changelog, message, tags)
    except ReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\n".join(decision.lines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
