"""CLI subcommand for grader calibration analysis.

Usage:
    eval-kit calibrate \
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
from pathlib import Path

from eval_kit.calibration.analyzer import (
    AnnotationSet,
    CalibrationAnalyzer,
)
from eval_kit.core.outcome import Outcome
from eval_kit.core.task import JSONTaskLoader
from eval_kit.core.transcript import Transcript
from eval_kit.execution.registry import load_class


def add_calibrate_parser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Add the 'calibrate' subcommand to the CLI."""
    parser = subparsers.add_parser(
        "calibrate",
        help="Check grader calibration against human annotations",
    )
    parser.add_argument(
        "--grader", required=True,
        help="Dotted path to Grader class",
    )
    parser.add_argument(
        "--samples", required=True,
        help="Path to eval set / samples JSON file",
    )
    parser.add_argument(
        "--annotations", required=True,
        help="Path to human annotations JSON file",
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
    """Execute the 'calibrate' subcommand."""
    # Load annotations
    with open(args.annotations) as f:
        annotations_data = json.load(f)
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
        # Grade transcripts on the fly
        grader_cls = load_class(args.grader)
        grader = grader_cls()

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
                    continue
                outcome = await grader.grade(transcript, task)
                outcomes[task_id] = outcome
            return outcomes

        grader_outcomes = asyncio.run(_grade_all())
    else:
        print("Error: provide either --transcripts or --results")
        return 1

    # Analyze calibration
    analyzer = CalibrationAnalyzer(threshold=args.threshold)
    result = analyzer.analyze(grader_outcomes, annotations)

    # Output
    print(result.render_table())

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    return 0 if result.is_calibrated else 1
