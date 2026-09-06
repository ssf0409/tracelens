"""Expected-error reporting shared by every subcommand (exit-code contract)."""

from __future__ import annotations

import argparse
import os
import sys
import traceback


def debug_enabled(args: argparse.Namespace) -> bool:
    """Whether tracebacks should accompany expected-error messages."""
    return bool(getattr(args, "debug", False)) or bool(os.environ.get("TRACELENS_DEBUG"))


def usage_error(
    message: str,
    *,
    hint: str | None = None,
    exc: BaseException | None = None,
    debug: bool = False,
) -> int:
    """Print a concise usage/input error to stderr and return exit code 2.

    Expected errors (a missing file, an unimportable class, an invalid value)
    are reported in one or two lines. The traceback is printed only with
    ``--debug`` / ``TRACELENS_DEBUG=1`` so a misconfiguration never looks like
    a crash, while a genuine programming failure stays diagnosable.
    """
    print(f"Error: {message}", file=sys.stderr)
    if hint:
        print(f"  {hint}", file=sys.stderr)
    if exc is not None:
        if debug:
            traceback.print_exception(exc, file=sys.stderr)
        else:
            print("  (run with --debug for the full traceback)", file=sys.stderr)
    return 2
