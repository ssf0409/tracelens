"""CLI entry point for tracelens.

Usage:
    tracelens run --eval-set tasks.json --adapter my.Adapter --graders my.Grader1 my.Grader2
    tracelens report --results results.json --format markdown
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from pathlib import Path

from pydantic import ValidationError

from tracelens.baselines.comparison import (
    DEFAULT_NOISE_BAND_ABSOLUTE,
    RegressionSeverity,
)
from tracelens.baselines.manager import BaselineManager
from tracelens.cli.calibrate import add_calibrate_parser, cmd_calibrate
from tracelens.cli.init import add_init_parser, cmd_init
from tracelens.cli.sample import add_sample_parser, cmd_sample
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.task import EvalSet
from tracelens.core.trial import Trial
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.registry import load_class
from tracelens.execution.runner import (
    DEFAULT_INFRA_EXCEPTION_TYPES,
    CheckpointError,
    EvaluationRunner,
    RunnerConfig,
)
from tracelens.loaders import EVAL_SET_FORMATS, EvalSetLoadError, load_tasks
from tracelens.reporting.gate import (
    GateResult,
    GateStatus,
    TaskGateOutcome,
    evaluate_gate,
    per_trial_results,
    spec_from_trials,
)
from tracelens.reporting.generator import ReportData, ReportGenerator

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for tracelens CLI."""
    parser = argparse.ArgumentParser(
        prog="tracelens",
        description="Evaluation framework for AI agents",
        epilog=(
            "Exit codes: 0 = success (or the gate passed); 1 = a negative result "
            "(a blocked gate, a calibration below threshold); 2 = a usage, "
            "configuration, or input error, or a gate that could not be "
            "evaluated. Set TRACELENS_DEBUG=1 or pass --debug for tracebacks."
        ),
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print full tracebacks for input and configuration errors",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- tracelens run --
    run_parser = subparsers.add_parser("run", help="Run an evaluation suite")
    run_parser.add_argument(
        "--eval-set", required=True,
        help=(
            "Path to the eval set: a .json, .jsonl, or .csv file, or a "
            "directory (then pass --eval-set-format)"
        ),
    )
    run_parser.add_argument(
        "--eval-set-format", choices=EVAL_SET_FORMATS, default=None,
        help=(
            "Format of --eval-set; inferred from the file suffix, required "
            "for a directory"
        ),
    )
    run_parser.add_argument(
        "--input-field", default="input",
        help=(
            "jsonl/csv eval sets: name of the column holding the task input "
            "(default: input)"
        ),
    )
    run_parser.add_argument(
        "--metadata-fields", nargs="+", default=None, metavar="FIELD",
        help=(
            "jsonl/csv eval sets: foreign columns to keep in Task.metadata "
            "(default: all of them)"
        ),
    )
    run_parser.add_argument(
        "--adapter", required=True,
        help="Dotted path to AgentAdapter class",
    )
    run_parser.add_argument(
        "--graders", required=True, nargs="+",
        help="Dotted paths to Grader classes",
    )
    run_parser.add_argument(
        "--num-runs", type=int, default=1,
        help="Number of runs per task (default: 1)",
    )
    run_parser.add_argument(
        "--max-concurrency", type=int, default=5,
        help="Max concurrent trials (default: 5)",
    )
    run_parser.add_argument(
        "--timeout", type=float, default=300.0,
        help="Timeout per trial in seconds (default: 300)",
    )
    run_parser.add_argument(
        "--baseline-check", action="store_true",
        help=(
            "Check results against baselines. Requires --baselines-file; "
            "a missing flag or file is a usage error (exit 2)"
        ),
    )
    run_parser.add_argument(
        "--baselines-file", default=None,
        help="Path to baselines JSON file",
    )
    run_parser.add_argument(
        "--require-baselines", action="store_true",
        help=(
            "Fail (exit 1) if any task in the eval set has no stored "
            "baseline, instead of warning and skipping it"
        ),
    )
    run_parser.add_argument(
        "--fail-on-regression", default="moderate",
        choices=["minor", "moderate", "severe"],
        help="Minimum regression severity to fail (default: moderate)",
    )
    run_parser.add_argument(
        "--output", default=None,
        help="Path to write JSON results",
    )
    run_parser.add_argument(
        "--report", default=None,
        help="Path to write markdown report",
    )
    run_parser.add_argument(
        "--html-report", default=None,
        help="Path to write HTML dashboard report",
    )
    run_parser.add_argument(
        "--save-trials", default=None,
        help="Path to write raw trial data (JSON) for replay and comparison",
    )
    run_parser.add_argument(
        "--progress", action="store_true",
        help="Print per-trial progress to stderr",
    )
    run_parser.add_argument(
        "--checkpoint", default=None,
        help=(
            "Path to a checkpoint file. Trials are periodically persisted "
            "there; re-running with the same path resumes, skipping "
            "already-completed trials (infra-errored trials re-run)"
        ),
    )
    run_parser.add_argument(
        "--max-infra-retries", type=int, default=0,
        help=(
            "Re-attempt trials that end in INFRA_ERROR up to N extra times "
            "with exponential backoff. Agent failures and timeouts never "
            "retry (default: 0)"
        ),
    )
    run_parser.add_argument(
        "--infra-exceptions", default=None, nargs="+",
        help=(
            "Dotted paths of extra exception types to classify as "
            "INFRA_ERROR instead of FAILED (e.g. builtins.OSError "
            "myproject.errors.RateLimitError). Extends the conservative "
            "default set (InfraError, MemoryError, ConnectionError)"
        ),
    )
    run_parser.add_argument(
        "--decision-spec", default=None,
        help=(
            "Path to a DecisionSpec JSON file describing this run's "
            "configuration. Stamped onto transcripts and, together with a "
            "baseline that carries its own spec, enables infra-noise-aware "
            "regression comparison in --baseline-check"
        ),
    )
    run_parser.add_argument(
        "--noise-band", type=float, default=None,
        help=(
            "Absolute metric delta treated as within infra noise when "
            f"baseline and current infra configs differ (default: "
            f"{DEFAULT_NOISE_BAND_ABSOLUTE}, i.e. 3 percentage points on "
            "a 0-1 metric). Requires --baseline-check"
        ),
    )

    # -- tracelens report --
    report_parser = subparsers.add_parser(
        "report", help="Generate report from results",
    )
    report_parser.add_argument(
        "--results", required=True,
        help="Path to JSON results file",
    )
    report_parser.add_argument(
        "--format", default="markdown", choices=["markdown", "json", "html"],
        help="Output format (default: markdown)",
    )

    # -- tracelens sample --
    add_sample_parser(subparsers)

    # -- tracelens init --
    add_init_parser(subparsers)

    # -- tracelens calibrate (and its 'reconcile' alias) --
    add_calibrate_parser(subparsers)
    add_calibrate_parser(subparsers, name="reconcile")

    return parser


def _severity_from_str(s: str) -> RegressionSeverity:
    return RegressionSeverity(s)


def _load_infra_exceptions(
    paths: list[str],
) -> tuple[type[BaseException], ...] | None:
    """Load --infra-exceptions dotted paths into exception types.

    Returns None (with a message on stderr) if any path fails to import
    or names something that isn't an exception type.
    """
    types: list[type[BaseException]] = []
    for path in paths:
        try:
            cls = load_class(path)
        except (ImportError, AttributeError) as exc:
            print(
                f"Error: could not load infra exception '{path}': {exc}",
                file=sys.stderr,
            )
            return None
        if not (isinstance(cls, type) and issubclass(cls, BaseException)):
            print(
                f"Error: '{path}' is not an exception type",
                file=sys.stderr,
            )
            return None
        types.append(cls)
    return tuple(types)


def _spec_from_trials(trials: list[Trial]) -> DecisionSpec | None:
    """Recover the run's DecisionSpec from adapter-stamped transcripts.

    Thin CLI wrapper over :func:`tracelens.reporting.gate.spec_from_trials`
    that prints the mixed-spec warning to stderr.
    """
    spec, warning = spec_from_trials(trials)
    if warning:
        print(f"[tracelens] warning: {warning}", file=sys.stderr)
    return spec


# One metric sample per gradable trial; see tracelens.reporting.gate.
_per_trial_results = per_trial_results


def _print_gate_diagnostics(gate: GateResult) -> None:
    """Per-task notes for a gate run: warnings to stderr, notes to stdout.

    The decision itself is rendered by ``ReportGenerator.render_ci_summary``
    from the same ``GateResult`` that the JSON/Markdown/HTML outputs carry.
    """
    for warning in gate.warnings:
        print(f"[tracelens] warning: {warning}", file=sys.stderr)
    for task in gate.tasks:
        if task.outcome is TaskGateOutcome.NO_BASELINE:
            print(
                f"[tracelens] warning: no baseline for task "
                f"'{task.task_id}' — skipped in baseline check",
                file=sys.stderr,
            )
            continue
        if task.excluded_trials:
            print(
                f"[tracelens] note: excluded {task.excluded_trials} infra-error/"
                f"grader-error trial(s) from the baseline comparison "
                f"for task '{task.task_id}'",
                file=sys.stderr,
            )
        if task.outcome is TaskGateOutcome.NO_GRADABLE_TRIALS:
            print(
                f"[tracelens] warning: no gradable trials for task "
                f"'{task.task_id}' (all infra/grader failures) "
                f"— skipped in baseline check",
                file=sys.stderr,
            )
        elif task.outcome is TaskGateOutcome.NO_COMPARABLE_METRICS:
            print(
                f"[tracelens] warning: no comparable metrics for task "
                f"'{task.task_id}'; store a baseline for at least "
                f"one CLI metric: {', '.join(task.available_metrics)}",
                file=sys.stderr,
            )
        elif task.infra_config_mismatch:
            diff = ", ".join(
                f"{key}: {b} -> {c}"
                for key, (b, c) in sorted(task.infra_config_diff.items())
            )
            print(
                f"[tracelens] note: infra config mismatch vs baseline for "
                f"task '{task.task_id}' ({diff}); regressions "
                f"within the {gate.noise_band} noise band "
                f"are flagged but not blocking"
            )


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


def _write_output(path: str, content: str) -> None:
    """Write one output file, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def _validate_run_parameters(args: argparse.Namespace) -> str | None:
    """Return a usage-error message for an impossible run configuration."""
    if args.num_runs < 1:
        return f"--num-runs must be at least 1 (got {args.num_runs})"
    if args.max_concurrency < 1:
        return f"--max-concurrency must be at least 1 (got {args.max_concurrency})"
    if args.timeout <= 0:
        return f"--timeout must be a positive number of seconds (got {args.timeout})"
    if args.max_infra_retries < 0:
        return f"--max-infra-retries cannot be negative (got {args.max_infra_retries})"
    return None


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    debug = debug_enabled(args)
    invalid = _validate_run_parameters(args)
    if invalid:
        return usage_error(invalid)

    # Gate preflight — a misconfigured baseline check must fail before any
    # eval time is spent, never silently skip (exit 2 = usage error, so CI
    # can tell "misconfigured gate" apart from "gate blocked" exit 1).
    baseline_manager: BaselineManager | None = None
    if args.baseline_check:
        if not args.baselines_file:
            print(
                "Error: --baseline-check requires --baselines-file; "
                "refusing to run with a vacuously-passing gate",
                file=sys.stderr,
            )
            return 2
        if not Path(args.baselines_file).exists():
            print(
                f"Error: baselines file not found: {args.baselines_file}",
                file=sys.stderr,
            )
            return 2
        try:
            baseline_manager = BaselineManager(args.baselines_file)
        except (ValueError, KeyError, TypeError) as exc:
            print(
                f"Error: could not load baselines file "
                f"{args.baselines_file}: {exc}",
                file=sys.stderr,
            )
            return 2
    else:
        gate_only_flags = [
            flag
            for flag, is_set in (
                ("--require-baselines", args.require_baselines),
                ("--noise-band", args.noise_band is not None),
            )
            if is_set
        ]
        if gate_only_flags:
            print(
                f"Error: {', '.join(gate_only_flags)} require(s) "
                "--baseline-check; refusing to run with a "
                "vacuously-passing gate",
                file=sys.stderr,
            )
            return 2
        if args.baselines_file:
            print(
                "[tracelens] warning: --baselines-file has "
                "no effect without --baseline-check",
                file=sys.stderr,
            )

    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # Resolve --infra-exceptions before running (usage error -> exit 2)
    infra_exception_types = DEFAULT_INFRA_EXCEPTION_TYPES
    if args.infra_exceptions:
        extra_types = _load_infra_exceptions(args.infra_exceptions)
        if extra_types is None:
            return 2
        infra_exception_types = infra_exception_types + extra_types

    # Resolve --decision-spec before running (usage error -> exit 2)
    decision_spec: DecisionSpec | None = None
    if args.decision_spec:
        spec_path = Path(args.decision_spec)
        if not spec_path.exists():
            print(
                f"Error: --decision-spec file not found: {args.decision_spec}",
                file=sys.stderr,
            )
            return 2
        try:
            decision_spec = DecisionSpec.model_validate(
                json.loads(spec_path.read_text())
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            print(
                f"Error: invalid --decision-spec file {args.decision_spec}: {exc}",
                file=sys.stderr,
            )
            return 2

    # Load eval set (usage error -> exit 2, before any agent call)
    try:
        tasks = load_tasks(
            args.eval_set,
            format=args.eval_set_format,
            input_field=args.input_field,
            metadata_fields=args.metadata_fields,
        )
    except EvalSetLoadError as exc:
        return usage_error(str(exc), exc=exc, debug=debug)
    eval_set = EvalSet(name=Path(args.eval_set).stem, tasks=tasks)

    # Load adapter and graders (usage error -> exit 2, before any agent call)
    import_hint = (
        "Use a dotted path (package.module.ClassName) and run from the project "
        "root so the module is importable; the class must accept no constructor "
        "arguments."
    )
    try:
        adapter_cls = load_class(args.adapter)
        adapter: AgentAdapter = adapter_cls()
    except (ImportError, AttributeError, TypeError) as exc:
        return usage_error(
            f"could not load adapter '{args.adapter}': {exc}",
            hint=import_hint, exc=exc, debug=debug,
        )

    graders = []
    for grader_path in args.graders:
        try:
            grader_cls = load_class(grader_path)
            graders.append(grader_cls())
        except (ImportError, AttributeError, TypeError) as exc:
            return usage_error(
                f"could not load grader '{grader_path}': {exc}",
                hint=import_hint, exc=exc, debug=debug,
            )

    # Build runner config
    def _print_progress(done: int, total: int) -> None:
        print(f"[tracelens] {done}/{total} trials complete", file=sys.stderr)

    config = RunnerConfig(
        num_runs=args.num_runs,
        max_concurrency=args.max_concurrency,
        timeout_seconds=args.timeout,
        infra_exception_types=infra_exception_types,
        progress_callback=_print_progress if args.progress else None,
        checkpoint_path=args.checkpoint,
        max_infra_retries=args.max_infra_retries,
    )

    # Run evaluation
    runner = EvaluationRunner(adapter, graders, config, decision_spec=decision_spec)
    try:
        batch = asyncio.run(runner.run(eval_set))
    except CheckpointError as exc:
        # A stale/corrupt/foreign checkpoint is a misconfigured run, not a
        # blocked gate — same exit-2 contract as the gate preflight.
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Generate report
    gen = ReportGenerator()
    report = gen.build_report(batch)

    # Baseline gate: one decision, attached to the report before any output
    # is written, so the exit code, stdout, JSON, Markdown, and HTML agree.
    if args.baseline_check and baseline_manager is not None:
        gate = evaluate_gate(
            batch,
            baseline_manager,
            threshold=_severity_from_str(args.fail_on_regression),
            noise_band=(
                args.noise_band
                if args.noise_band is not None
                else DEFAULT_NOISE_BAND_ABSOLUTE
            ),
            require_baselines=args.require_baselines,
            decision_spec=decision_spec,
            task_ids=[summary.task_id for summary in report.task_summaries],
        )
        _print_gate_diagnostics(gate)
    else:
        gate = GateResult.not_requested()
    report.gate = gate

    # Write outputs (always, even when the gate blocks: the artifacts are
    # the evidence). A write failure is a clear error, never a traceback.
    written: list[tuple[str, str]] = []
    try:
        if args.output:
            _write_output(args.output, json.dumps(report.to_dict(), indent=2, default=str))
            written.append(("results", args.output))
        if args.report:
            _write_output(args.report, gen.render_markdown(report))
            written.append(("report", args.report))
        if args.html_report:
            _write_output(args.html_report, gen.render_html(report))
            written.append(("html report", args.html_report))
        if args.save_trials:
            _write_output(args.save_trials, json.dumps(batch.to_dict(), indent=2))
            written.append(("trials", args.save_trials))
    except OSError as exc:
        return usage_error(f"could not write output file: {exc}", exc=exc, debug=debug)
    # Say where the artifacts went (stderr: stdout stays the summary only).
    for label, path in written:
        print(f"[tracelens] wrote {label}: {path}", file=sys.stderr)
    if args.checkpoint:
        print(f"[tracelens] checkpoint: {args.checkpoint}", file=sys.stderr)

    # CI summary to stdout, including the gate lines when a gate ran
    print(gen.render_ci_summary(report))

    if gate.status is GateStatus.UNEVALUABLE:
        # Missing evidence invalidates the check even if another task regressed.
        print(
            "Error: baseline check is unevaluable; verify the eval set, "
            "run count, and matching baseline metrics; fix any infra/grader "
            "failures listed above and rerun. This is not a passing gate.",
            file=sys.stderr,
        )
        return 2
    if gate.status is GateStatus.BLOCKED:
        for reason in gate.reasons:
            print(f"Error: {reason}", file=sys.stderr)
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Execute the 'report' subcommand.

    Re-renders a results file written by ``tracelens run --output``,
    including its recorded baseline gate decision. Input problems (missing
    file, invalid JSON, not a results document) exit 2.
    """
    debug = debug_enabled(args)
    try:
        data = json.loads(Path(args.results).read_text())
    except FileNotFoundError as exc:
        return usage_error(f"results file not found: {args.results}", exc=exc, debug=debug)
    except json.JSONDecodeError as exc:
        return usage_error(
            f"invalid JSON in results file {args.results}: {exc}", exc=exc, debug=debug
        )
    try:
        report = ReportData.from_dict(data)
    except ValueError as exc:
        return usage_error(
            f"{args.results} is not a TraceLens results file ({exc})",
            hint="Pass the JSON written by 'tracelens run --output'.",
            exc=exc, debug=debug,
        )
    gen = ReportGenerator()

    if args.format == "markdown":
        print(gen.render_markdown(report))
    elif args.format == "html":
        print(gen.render_html(report))
    else:
        print(json.dumps(report.to_dict(), indent=2, default=str))

    return 0


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        sys.exit(cmd_run(args))
    elif args.command == "report":
        sys.exit(cmd_report(args))
    elif args.command == "sample":
        sys.exit(cmd_sample(args))
    elif args.command == "init":
        sys.exit(cmd_init(args))
    elif args.command in ("calibrate", "reconcile"):
        sys.exit(cmd_calibrate(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
