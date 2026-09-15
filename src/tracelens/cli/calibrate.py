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

from pydantic import ValidationError

from tracelens.calibration.analyzer import (
    AnnotationSet,
    CalibrationAnalyzer,
    CalibrationResult,
)
from tracelens.cli._errors import debug_enabled, usage_error
from tracelens.core.outcome import Outcome
from tracelens.core.transcript import Transcript
from tracelens.core.trial import TrialBatch
from tracelens.execution.registry import load_class
from tracelens.loaders import EvalSetLoadError, load_tasks


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
        help="Path to transcripts JSON file (trials.json or dict of task_id → transcript)",
    )
    parser.add_argument(
        "--results",
        help="Path to results JSON file (dict of task_id → outcome or run results.json)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.7,
        help="Minimum Pearson r for calibrated (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        help="Path to write calibration result JSON",
    )
    parser.add_argument(
        "--import-root",
        help="Root directory for Python imports (defaults to current working directory)",
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Execute the 'calibrate' / 'reconcile' subcommand."""
    debug = debug_enabled(args)

    # Ensure import root is on sys.path so local project classes can be imported
    import_root_path = Path(args.import_root).resolve() if getattr(args, "import_root", None) else Path.cwd()
    import_root_str = str(import_root_path)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

    # Load annotations (input problems are usage errors: exit 2)
    try:
        with open(args.annotations) as f:
            annotations_data = json.load(f)
    except FileNotFoundError as exc:
        return usage_error(f"annotations file not found: {args.annotations}", exc=exc, debug=debug)
    except json.JSONDecodeError as exc:
        return usage_error(f"invalid JSON in {args.annotations}: {exc}", exc=exc, debug=debug)

    analyzer = CalibrationAnalyzer(threshold=args.threshold)

    # Self-contained review worksheet: each row carries the grader outcome next
    # to the human grade, so no separate --results/--transcripts is needed. This
    # is what `tracelens sample` produces and the documented default flow.
    if not args.results and not args.transcripts:
        try:
            result = analyzer.analyze_worksheet(annotations_data)
        except Exception as exc:
            return usage_error(f"could not parse worksheet in {args.annotations}: {exc}", exc=exc, debug=debug)

        if result.sample_count == 0:
            return usage_error(
                f"no usable rows in {args.annotations}. Expected a filled "
                f"review worksheet (rows with grader_score + human_score), or pass "
                f"--results / --transcripts."
            )
        return _emit_calibration(result, args)

    try:
        annotations = AnnotationSet.from_json_list(annotations_data)
    except (TypeError, ValueError, ValidationError) as exc:
        return usage_error(
            f"invalid annotations format in {args.annotations}: {exc}",
            exc=exc,
            debug=debug,
        )

    # Determine grader outcomes
    grader_outcomes: dict[str, Outcome] = {}

    if args.results:
        # Load pre-computed results
        try:
            with open(args.results) as f:
                results_data = json.load(f)
        except FileNotFoundError as exc:
            return usage_error(f"results file not found: {args.results}", exc=exc, debug=debug)
        except json.JSONDecodeError as exc:
            return usage_error(f"invalid JSON in {args.results}: {exc}", exc=exc, debug=debug)

        if not isinstance(results_data, dict):
            return usage_error(
                f"expected a JSON object in results file {args.results}",
                hint="Expected {task_id: outcome} or report results.json format",
            )

        # Support results from `tracelens run --output results.json` (report document)
        if "task_summaries" in results_data and isinstance(results_data["task_summaries"], list):
            for summary in results_data["task_summaries"]:
                tid = summary.get("task_id")
                if tid:
                    score = float(summary.get("mean_score", 0.0))
                    passed = bool(summary.get("pass_rate", 0.0) >= 0.5)
                    grader_outcomes[tid] = Outcome(
                        trial_id=f"run-{tid}",
                        grader_id="run-summary",
                        passed=passed,
                        score=score,
                    )
        else:
            for task_id, outcome_data in results_data.items():
                if not isinstance(outcome_data, dict):
                    return usage_error(
                        f"invalid outcome data for task '{task_id}' in {args.results}: expected a JSON object",
                    )
                try:
                    grader_outcomes[task_id] = Outcome(**outcome_data)
                except (TypeError, ValueError, ValidationError) as exc:
                    return usage_error(
                        f"invalid outcome format for task '{task_id}' in {args.results}: {exc}",
                        exc=exc,
                        debug=debug,
                    )
    elif args.transcripts:
        if not args.grader or not args.samples:
            return usage_error("--transcripts requires --grader and --samples")

        # Grade transcripts on the fly
        try:
            grader_cls = load_class(args.grader)
        except (ImportError, AttributeError) as exc:
            return usage_error(
                f"could not load grader '{args.grader}': {exc}",
                hint=f"Modules are imported from {import_root_str}.",
                exc=exc,
                debug=debug,
            )

        grader_id = args.grader.rsplit(".", 1)[-1]
        try:
            grader = grader_cls()
        except TypeError:
            try:
                grader = grader_cls(grader_id)
            except Exception as exc:
                return usage_error(f"could not instantiate grader '{args.grader}': {exc}", exc=exc, debug=debug)

        try:
            with open(args.transcripts) as f:
                transcripts_json = json.load(f)
        except FileNotFoundError as exc:
            return usage_error(f"transcripts file not found: {args.transcripts}", exc=exc, debug=debug)
        except json.JSONDecodeError as exc:
            return usage_error(f"invalid JSON in {args.transcripts}: {exc}", exc=exc, debug=debug)

        try:
            tasks = load_tasks(args.samples)
        except FileNotFoundError as exc:
            return usage_error(f"samples file not found: {args.samples}", exc=exc, debug=debug)
        except (EvalSetLoadError, json.JSONDecodeError, ValidationError) as exc:
            return usage_error(f"could not load samples from '{args.samples}': {exc}", exc=exc, debug=debug)

        task_map = {t.task_id: t for t in tasks}

        # Extract transcripts: support TrialBatch (trials.json) and dict mapping {task_id: transcript}
        transcripts_to_grade: dict[str, Transcript] = {}
        if isinstance(transcripts_json, dict) and "trials" in transcripts_json:
            try:
                batch = TrialBatch.from_dict(transcripts_json)
                for t in batch.trials:
                    if t.transcript is not None:
                        transcripts_to_grade[t.task_id] = t.transcript
            except Exception as exc:
                return usage_error(f"failed to parse trials batch from {args.transcripts}: {exc}", exc=exc, debug=debug)
        elif isinstance(transcripts_json, dict):
            for task_id, tdata in transcripts_json.items():
                if isinstance(tdata, dict):
                    try:
                        transcripts_to_grade[task_id] = Transcript(**tdata)
                    except Exception as exc:
                        return usage_error(f"invalid transcript for '{task_id}' in {args.transcripts}: {exc}", exc=exc, debug=debug)
                else:
                    return usage_error(f"transcript for '{task_id}' in {args.transcripts} must be an object")
        else:
            return usage_error(f"unexpected format in {args.transcripts}: expected a trials JSON batch or transcript map")

        async def _grade_all() -> dict[str, Outcome]:
            outcomes: dict[str, Outcome] = {}
            for task_id, transcript in transcripts_to_grade.items():
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

    if result.status == "NOT EVALUABLE (constant scores)":
        return 2
    return 0 if result.is_calibrated else 1
