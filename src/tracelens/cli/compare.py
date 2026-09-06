"""CLI subcommand to compare two saved runs (issue #28).

Usage:
    tracelens compare baseline-trials.json candidate-trials.json \\
        --metric pass_rate --threshold 0.03 --output compare.json

Both inputs are the artifacts written by ``tracelens run --save-trials``.
The statistics follow the "Run-versus-run comparison" section of
``docs/statistical-contract.md``: tasks are aligned by content through the
runs' provenance, one statistic per task and run is paired, and the mean
paired difference is reported with a task bootstrap interval, a sign-flip
p-value, and a verdict against a practical threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tracelens.cli._errors import debug_enabled, usage_error
from tracelens.core.trial import TrialBatch
from tracelens.statistics.run_comparison import (
    DEFAULT_THRESHOLD,
    UNMATCHED_POLICIES,
    ComparisonError,
    compare_runs,
)


def add_compare_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the 'compare' subcommand to the CLI."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare two saved runs with a paired task bootstrap",
        description=(
            "Decide whether a candidate run is better, worse, or indistinguishable "
            "from a baseline run of the same eval set. Inputs are the trials files "
            "written by 'tracelens run --save-trials'."
        ),
        epilog=(
            "Exit codes: 0 = evaluated with no regression (improvement, significant "
            "but below the threshold, or equivalent within it); 1 = regression; "
            "2 = unevaluable (incompatible runs, insufficient evidence, inconclusive, "
            "or an input error). --observe exits 0 for every evaluated comparison."
        ),
    )
    parser.add_argument("baseline", help="Trials JSON of the reference run")
    parser.add_argument("candidate", help="Trials JSON of the run under test")
    parser.add_argument(
        "--metric", default="pass_rate", metavar="METRIC",
        help=(
            "pass_rate (default), mean_score, or <grader_id>.<metric_name> for an "
            "outcome metric"
        ),
    )
    parser.add_argument(
        "--direction", choices=["higher", "lower"], default=None,
        help="Which way is better for a <grader_id>.<metric_name> metric (default: higher)",
    )
    parser.add_argument(
        "--grader", default=None, metavar="GRADER_ID",
        help="Restrict pass_rate / mean_score to one grader's outcome",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=(
            "Practical threshold: an absolute delta on the metric's scale "
            f"(default: {DEFAULT_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level of the interval (default: 0.95)",
    )
    parser.add_argument(
        "--bootstrap", type=int, default=10000, dest="n_bootstrap", metavar="B",
        help="Bootstrap resamples and sign-flip draws (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the resampling; same inputs and seed reproduce the result (default: 0)",
    )
    parser.add_argument(
        "--unmatched-tasks", choices=list(UNMATCHED_POLICIES), default="error",
        dest="unmatched_tasks",
        help=(
            "What to do when the task sets differ: 'error' refuses (default); "
            "'exclude' compares the shared, unchanged tasks and lists the rest"
        ),
    )
    parser.add_argument(
        "--require-provenance", action="store_true", dest="require_provenance",
        help="Refuse artifacts without provenance instead of aligning tasks by id",
    )
    parser.add_argument(
        "--observe", action="store_true",
        help="Observational mode: exit 0 for every evaluated comparison",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="How many per-task movers to print (default: 5)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write the comparison JSON (same fields as the summary)",
    )


def _load_batch(path: str, *, debug: bool) -> TrialBatch | int:
    """Load a trials artifact, or return the exit code of a usage error."""
    try:
        with open(path, encoding="utf-8") as handle:
            data: Any = json.load(handle)
    except FileNotFoundError:
        return usage_error(f"trials file not found: {path}")
    except json.JSONDecodeError as exc:
        return usage_error(f"invalid JSON in {path}: {exc}", exc=exc, debug=debug)
    if isinstance(data, dict) and "trials" not in data and "task_summaries" in data:
        return usage_error(
            f"{path} is a results file (tracelens run --output); it has no per-trial "
            "samples",
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


def cmd_compare(args: argparse.Namespace) -> int:
    """Execute the 'compare' subcommand."""
    debug = debug_enabled(args)
    baseline = _load_batch(args.baseline, debug=debug)
    if isinstance(baseline, int):
        return baseline
    candidate = _load_batch(args.candidate, debug=debug)
    if isinstance(candidate, int):
        return candidate
    try:
        result = compare_runs(
            baseline,
            candidate,
            metric=args.metric,
            direction=args.direction,
            grader=args.grader,
            threshold=args.threshold,
            confidence=args.confidence,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            unmatched_tasks=args.unmatched_tasks,
            require_provenance=args.require_provenance,
            observe=args.observe,
            baseline_label=Path(args.baseline).name,
            candidate_label=Path(args.candidate).name,
        )
    except ComparisonError as exc:
        return usage_error(str(exc))

    print("\n".join(result.summary_lines(top=args.top)))
    if args.output:
        target = Path(args.output)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        except OSError as exc:
            return usage_error(f"could not write output file: {exc}", exc=exc, debug=debug)
        print(f"[tracelens] wrote comparison: {args.output}", file=sys.stderr)
    return result.exit_code
