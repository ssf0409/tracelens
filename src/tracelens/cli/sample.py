"""CLI subcommand to select trials for human review.

Usage:
    tracelens sample \
        --trials trials.json \
        --size 20 \
        --strategy diverse \
        --output review.json

`trials.json` is the artifact written by ``tracelens run --save-trials``. The
emitted ``review.json`` is a worksheet: a reviewer fills in ``human_score`` and
``human_passed`` for each row, then feeds it to ``tracelens reconcile
--annotations review.json`` to measure grader/human agreement.
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from tracelens.calibration.sampler import STRATEGIES, sample_for_review
from tracelens.core.trial import TrialBatch


def add_sample_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the 'sample' subcommand to the CLI."""
    parser = subparsers.add_parser(
        "sample",
        help="Select trials for human review (feeds 'reconcile')",
    )
    parser.add_argument(
        "--trials", required=True,
        help="Path to trials JSON from 'tracelens run --save-trials'",
    )
    parser.add_argument(
        "--size", type=int, default=20,
        help="Number of trials to select (default: 20)",
    )
    parser.add_argument(
        "--strategy", default="diverse", choices=list(STRATEGIES),
        help="Selection strategy (default: diverse)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the 'random' strategy (default: 0)",
    )
    parser.add_argument(
        "--excerpt-chars", type=int, default=280, dest="excerpt_chars",
        help="Max chars of each final output to include (default: 280)",
    )
    parser.add_argument(
        "--output",
        help="Path to write the review worksheet JSON (default: stdout)",
    )


def cmd_sample(args: argparse.Namespace) -> int:
    """Execute the 'sample' subcommand."""
    # Input problems are usage errors (exit 2), consistent with `tracelens run`.
    try:
        with open(args.trials) as f:
            batch = TrialBatch.from_dict(json.load(f))
    except FileNotFoundError:
        print(f"Error: trials file not found: {args.trials}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in {args.trials}: {exc}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(
            f"Error: {args.trials} is not a valid trials file "
            f"(expected 'tracelens run --save-trials' output): {exc}",
            file=sys.stderr,
        )
        return 2

    worksheet = sample_for_review(
        batch,
        size=args.size,
        strategy=args.strategy,
        seed=args.seed,
        excerpt_chars=args.excerpt_chars,
    )
    rows = worksheet.to_annotation_template()

    if not rows:
        print(
            f"Warning: no gradeable trials found in {args.trials} "
            f"(need trials with a transcript and a grader outcome). "
            f"Is this a 'tracelens run --save-trials' file?",
            file=sys.stderr,
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(
            f"Wrote {len(rows)} review items ({args.strategy}) to {args.output}",
            file=sys.stderr,
        )
    else:
        print(json.dumps(rows, indent=2))

    return 0
