"""CLI subcommand to explain failed trials from a saved artifact (issue #52).

Usage:
    tracelens inspect eval/results/trials.json --failures
    tracelens inspect trials.json --task-id t-fail --eval-set eval/tasks.json
    tracelens inspect trials.json --failures --html eval/results/failures.html

Reads the artifact written by ``tracelens run --save-trials`` and prints, per
selected trial, why it failed (agent, infra, or grader), what was expected
and what came back, each grader's verdict and feedback, and the transcript
steps. Output is bounded by default; omitted content is counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tracelens.cli._errors import debug_enabled, usage_error
from tracelens.core.task import Task
from tracelens.core.trial import TrialBatch
from tracelens.loaders import EVAL_SET_FORMATS, EvalSetLoadError, load_tasks
from tracelens.reporting.inspect import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_STEPS,
    FAILURE_KINDS,
    KIND_FLAGS,
    TrialKind,
    build_inspection,
    render_html,
    render_text,
)


def add_inspect_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the 'inspect' subcommand to the CLI."""
    parser = subparsers.add_parser(
        "inspect",
        help="Explain failed trials from a trials file",
        description=(
            "Explain the trials in a 'tracelens run --save-trials' file: why each "
            "failed (agent failure, infra error, or grader crash), what was expected "
            "and what came back, what each grader said, and what the transcript did. "
            "Shows failures by default; the exit code is 0 whenever the file could be "
            "read (this command reports, the gate decides)."
        ),
    )
    parser.add_argument("trials", help="Trials JSON from 'tracelens run --save-trials'")
    parser.add_argument(
        "--failures", action="store_true",
        help="Show agent failures, infra errors, and grader errors (the default)",
    )
    parser.add_argument(
        "--all", action="store_true", help="Show every trial, passed ones included",
    )
    parser.add_argument(
        "--kind", nargs="+", choices=list(KIND_FLAGS), default=None, metavar="KIND",
        help=(
            "Show only these kinds: agent, infra, grader, not-run, passed "
            "(overrides --failures/--all)"
        ),
    )
    parser.add_argument(
        "--task-id", nargs="+", default=None, dest="task_ids", metavar="ID",
        help="Show only these tasks",
    )
    parser.add_argument(
        "--grader", nargs="+", default=None, dest="grader_ids", metavar="GRADER_ID",
        help="Show only trials that these graders failed or crashed on",
    )
    parser.add_argument(
        "--eval-set", default=None, dest="eval_set",
        help="Eval set the run used; adds each task's name, input, and expected output",
    )
    parser.add_argument(
        "--eval-set-format", choices=EVAL_SET_FORMATS, default=None, dest="eval_set_format",
        help="Format of --eval-set (inferred from the suffix; required for a directory)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, dest="max_steps",
        help=f"Transcript steps to show per trial (default: {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--max-chars", type=int, default=DEFAULT_MAX_CHARS, dest="max_chars",
        help=f"Characters to show per field (default: {DEFAULT_MAX_CHARS})",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="No bounds: embed complete transcripts (may include sensitive content)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Show at most this many trials (the count of matches is always reported)",
    )
    parser.add_argument(
        "--html", default=None, metavar="PATH",
        help="Also write a self-contained HTML drilldown (works offline)",
    )
    parser.add_argument(
        "--json", default=None, metavar="PATH",
        help="Also write the inspection as JSON (the same fields the text shows)",
    )


def _load_batch(path: str, *, debug: bool) -> TrialBatch | int:
    try:
        with open(path, encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except FileNotFoundError:
        return usage_error(f"trials file not found: {path}")
    except json.JSONDecodeError as exc:
        return usage_error(f"invalid JSON in {path}: {exc}", exc=exc, debug=debug)
    if isinstance(data, dict) and "trials" not in data and "task_summaries" in data:
        return usage_error(
            f"{path} is a results file (tracelens run --output); it has no trials",
            hint="Pass the trials file written by 'tracelens run --save-trials'.",
        )
    try:
        return TrialBatch.from_dict(data)
    except ValidationError as exc:
        return usage_error(
            f"{path} is not a valid trials file (expected 'tracelens run --save-trials' "
            f"output): {exc}",
            exc=exc, debug=debug,
        )


def _write(path: str, content: str, label: str, *, debug: bool) -> int | None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return usage_error(f"could not write {label}: {exc}", exc=exc, debug=debug)
    print(f"[tracelens] wrote {label}: {path}", file=sys.stderr)
    return None


def cmd_inspect(args: argparse.Namespace) -> int:
    """Execute the 'inspect' subcommand."""
    debug = debug_enabled(args)
    for name, value in (("--max-steps", args.max_steps), ("--max-chars", args.max_chars)):
        if value < 0:
            return usage_error(f"{name} cannot be negative (got {value})")
    if args.limit is not None and args.limit < 1:
        return usage_error(f"--limit must be at least 1 (got {args.limit})")
    batch = _load_batch(args.trials, debug=debug)
    if isinstance(batch, int):
        return batch
    tasks: list[Task] | None = None
    if args.eval_set:
        try:
            tasks = load_tasks(args.eval_set, format=args.eval_set_format)
        except EvalSetLoadError as exc:
            return usage_error(str(exc), exc=exc, debug=debug)
    if args.kind:
        kinds: tuple[TrialKind, ...] | None = tuple(KIND_FLAGS[k] for k in args.kind)
    elif args.all:
        kinds = None
    else:
        kinds = FAILURE_KINDS
    report = build_inspection(
        batch,
        source=args.trials,
        kinds=kinds,
        task_ids=args.task_ids,
        grader_ids=args.grader_ids,
        tasks=tasks,
        max_steps=args.max_steps,
        max_chars=args.max_chars,
        full=args.full,
        limit=args.limit,
    )
    print(render_text(report))
    if args.html:
        failed = _write(args.html, render_html(report), "inspection html", debug=debug)
        if failed is not None:
            return failed
    if args.json:
        failed = _write(args.json, report.model_dump_json(indent=2), "inspection json", debug=debug)
        if failed is not None:
            return failed
    return 0
