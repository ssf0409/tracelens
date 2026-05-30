"""CLI subcommand for grader calibration analysis.

Usage:
    tracelens calibrate \
        --grader my.graders.QualityGrader \
        --samples eval/samples.json \
        --annotations human_grades.json \
        --transcripts transcripts.json \
        --threshold 0.7 \
        --output calibration.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from tracelens.calibration.analyzer import (
    AnnotationSet,
    CalibrationAnalyzer,
    CalibrationResult,
)
from tracelens.core.outcome import Outcome
from tracelens.core.task import JSONTaskLoader
from tracelens.core.transcript import Transcript
from tracelens.execution.registry import load_class


def add_calibrate_parser(
    subparsers: argparse._SubParsersAction,  # type: ignore[type-arg]
    name: str = "calibrate",
) -> None:
    """Add the 'calibrate' subcommand (and its 'reconcile' alias) to the CLI."""
    parser = subparsers.add_parser(
        name,
        help="Check grader calibration against human annotations",
    )
    parser.add_argument(
        "--grader",
        help="Dotted path to Grader class (required only with --transcripts)",
    )
    parser.add_argument(
        "--samples",
        help="Path to eval set / samples JSON file (required only with --transcripts)",
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to annotations / review-worksheet JSON file",
    )
    parser.add_argument(
        "--transcripts",
        help="Path to transcripts JSON file (dict of task_id → transcript)",
    )
    parser.add_argument(
        "--results",
        help="Path to results JSON file (dict of task_id → outcome)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Minimum Pearson r for calibrated (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        help="Path to write calibration result JSON",
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Execute the 'calibrate' / 'reconcile' subcommand."""
    # Load annotations
    with open(args.annotations) as f:
        annotations_data = json.load(f)

    analyzer = CalibrationAnalyzer(threshold=args.threshold)

    # Self-contained review worksheet: each row carries the grader outcome next
    # to the human grade, so no separate --results/--transcripts is needed. This
    # is what `tracelens sample` produces and the documented default flow.
    if not args.results and not args.transcripts:
        result = analyzer.analyze_worksheet(annotations_data)
        if result.sample_count == 0:
            print(
                f"Error: no usable rows in {args.annotations}. Expected a filled "
                f"review worksheet (rows with grader_score + human_score), or pass "
                f"--results / --transcripts.",
                file=sys.stderr,
            )
            return 1
        return _emit_calibration(result, args)

    annotations = AnnotationSet.from_json_list(annotations_data)

    # Determine grader outcomes
    grader_outcomes: dict[str, Outcome] = {}

    if args.results:
        # Load pre-computed results
        with open(args.results) as f:
            results_data = json.load(f)
        for task_id, outcome_data in results_data.items():
            grader_outcomes[task_id] = Outcome(**outcome_data)
    elif args.transcripts:
        if not args.grader or not args.samples:
            print(
                "Error: --transcripts requires --grader and --samples",
                file=sys.stderr,
            )
            return 1
        # Grade transcripts on the fly
        grader_cls = load_class(args.grader)
        grader_id = args.grader.rsplit(".", 1)[-1]
        grader = grader_cls(grader_id)

        with open(args.transcripts) as f:
            transcripts_data = json.load(f)

        loader = JSONTaskLoader()
        tasks = loader.load(args.samples)
        task_map = {t.task_id: t for t in tasks}

        async def _grade_all() -> dict[str, Outcome]:
            outcomes: dict[str, Outcome] = {}
            for task_id, transcript_data in transcripts_data.items():
                transcript = Transcript(**transcript_data)
                task = task_map.get(task_id)
                if task is None:
                    print(
                        f"Warning: transcript task_id '{task_id}' "
                        f"not found in samples file, skipping",
                        file=sys.stderr,
                    )
                    continue
                outcome = await grader.grade(transcript, task)
                outcomes[task_id] = outcome
            return outcomes

        grader_outcomes = asyncio.run(_grade_all())

    result = analyzer.analyze(grader_outcomes, annotations)
    return _emit_calibration(result, args)


def _emit_calibration(result: CalibrationResult, args: argparse.Namespace) -> int:
    """Print the calibration report, optionally write JSON, return exit code."""
    print(result.render_table())

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return 0 if result.is_calibrated else 1
