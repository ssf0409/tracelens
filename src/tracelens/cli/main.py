"""CLI entry point for tracelens.

Usage:
    tracelens run --eval-set tasks.json --adapter my.Adapter --graders my.Grader1 my.Grader2
    tracelens report --results results.json --format markdown
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from tracelens.baselines.comparison import (
    DEFAULT_NOISE_BAND_ABSOLUTE,
    RegressionDetector,
    RegressionSeverity,
)
from tracelens.baselines.manager import BaselineManager
from tracelens.cli.calibrate import add_calibrate_parser, cmd_calibrate
from tracelens.cli.init import add_init_parser, cmd_init
from tracelens.cli.sample import add_sample_parser, cmd_sample
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.task import EvalSet, JSONTaskLoader
from tracelens.core.trial import Trial, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.registry import load_class
from tracelens.execution.runner import (
    DEFAULT_INFRA_EXCEPTION_TYPES,
    CheckpointError,
    EvaluationRunner,
    RunnerConfig,
)
from tracelens.reporting.generator import ReportData, ReportGenerator

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for tracelens CLI."""
    parser = argparse.ArgumentParser(
        prog="tracelens",
        description="Evaluation framework for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- tracelens run --
    run_parser = subparsers.add_parser("run", help="Run an evaluation suite")
    run_parser.add_argument(
        "--eval-set", required=True,
        help="Path to eval set JSON file",
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
    """Recover the current run's DecisionSpec from adapter-stamped transcripts.

    Lets noise-aware comparison work without --decision-spec when the
    adapter records its own runtime configuration. The most recent spec
    wins: on checkpoint resume, trials from the previous run are loaded
    first, so the last-stamped spec belongs to the current run.
    """
    specs = [
        trial.transcript.decision_spec
        for trial in trials
        if trial.transcript is not None and trial.transcript.decision_spec is not None
    ]
    if not specs:
        return None
    if len({spec.fingerprint for spec in specs}) > 1:
        print(
            "[tracelens] warning: mixed decision specs found across trials "
            "(checkpoint resume with a changed config?); using the most "
            "recent — pass --decision-spec to be explicit",
            file=sys.stderr,
        )
    return specs[-1]


def _per_trial_results(trials: list[Trial]) -> list[dict[str, float]]:
    """One metric sample per gradable trial for regression detection.

    RegressionDetector.compare() runs a t-test over the sample
    distribution, so it needs per-trial values — a pre-aggregated
    single dict would collapse it to a one-sample z-test. The sample
    mean of the per-trial ``pass_rate`` indicators equals the task's
    pass rate, so baseline metric names stay unchanged.

    Trials that failed for harness reasons — INFRA_ERROR status or a
    grader crash — are excluded: they are surfaced separately via
    infra_error_rate / grader_error_rate and must not masquerade as
    agent regressions in the gate. TIMEOUT stays included: a run that
    blows the time budget is an agent-quality signal.
    """
    results: list[dict[str, float]] = []
    for trial in trials:
        if trial.status == TrialStatus.INFRA_ERROR or trial.has_grader_error:
            continue
        results.append({
            "pass_rate": 1.0 if trial.passed else 0.0,
            "mean_score": (
                trial.aggregate_score if trial.aggregate_score is not None else 0.0
            ),
        })
    return results


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
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

    # Load eval set
    try:
        loader = JSONTaskLoader()
        tasks = loader.load(args.eval_set)
        eval_set = EvalSet(name=Path(args.eval_set).stem, tasks=tasks)
    except FileNotFoundError:
        print(f"Error: eval-set file not found: {args.eval_set}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(
            f"Error: invalid JSON in eval-set file {args.eval_set}: {exc}",
            file=sys.stderr,
        )
        return 1

    # Load adapter and graders
    try:
        adapter_cls = load_class(args.adapter)
        adapter: AgentAdapter = adapter_cls()
    except (ImportError, AttributeError) as exc:
        print(
            f"Error: could not load adapter '{args.adapter}': {exc}",
            file=sys.stderr,
        )
        return 1

    graders = []
    for grader_path in args.graders:
        try:
            grader_cls = load_class(grader_path)
            graders.append(grader_cls())
        except (ImportError, AttributeError) as exc:
            print(
                f"Error: could not load grader '{grader_path}': {exc}",
                file=sys.stderr,
            )
            return 1

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

    # Write outputs
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            f.write(gen.render_markdown(report))

    if args.html_report:
        Path(args.html_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.html_report, "w") as f:
            f.write(gen.render_html(report))

    # Save raw trial data for replay and comparison
    if args.save_trials:
        Path(args.save_trials).parent.mkdir(parents=True, exist_ok=True)
        trial_data = batch.to_dict()
        with open(args.save_trials, "w") as f:
            json.dump(trial_data, f, indent=2)

    # CI summary to stdout
    print(gen.render_ci_summary(report))

    # Baseline check (manager loaded and validated in the preflight)
    if args.baseline_check and baseline_manager is not None:
        detector = RegressionDetector(
            noise_band_absolute=(
                args.noise_band
                if args.noise_band is not None
                else DEFAULT_NOISE_BAND_ABSOLUTE
            )
        )

        threshold = _severity_from_str(args.fail_on_regression)

        trials_by_task: dict[str, list[Trial]] = {}
        for trial in batch.trials:
            trials_by_task.setdefault(trial.task_id, []).append(trial)

        checked = 0
        skipped: list[str] = []
        no_gradable: list[str] = []
        no_comparable_metrics: list[str] = []
        blocking = 0
        for task_summary in report.task_summaries:
            baseline = baseline_manager.get_baseline(task_summary.task_id)
            if baseline is None:
                skipped.append(task_summary.task_id)
                print(
                    f"[tracelens] warning: no baseline for task "
                    f"'{task_summary.task_id}' — skipped in baseline check",
                    file=sys.stderr,
                )
                continue
            task_trials = trials_by_task.get(task_summary.task_id, [])
            current_results = _per_trial_results(task_trials)
            excluded = len(task_trials) - len(current_results)
            if excluded:
                print(
                    f"[tracelens] note: excluded {excluded} infra-error/"
                    f"grader-error trial(s) from the baseline comparison "
                    f"for task '{task_summary.task_id}'",
                    file=sys.stderr,
                )
            if not current_results:
                no_gradable.append(task_summary.task_id)
                print(
                    f"[tracelens] warning: no gradable trials for task "
                    f"'{task_summary.task_id}' (all infra/grader failures) "
                    f"— skipped in baseline check",
                    file=sys.stderr,
                )
                continue
            current_metrics = {name for result in current_results for name in result}
            if not baseline.metrics.keys() & current_metrics:
                no_comparable_metrics.append(task_summary.task_id)
                print(
                    f"[tracelens] warning: no comparable metrics for task "
                    f"'{task_summary.task_id}'; store a baseline for at least "
                    f"one CLI metric: {', '.join(sorted(current_metrics))}",
                    file=sys.stderr,
                )
                continue
            checked += 1
            current_spec = decision_spec or _spec_from_trials(task_trials)
            reg_report = detector.compare_with_specs(
                baseline,
                current_results,
                baseline_spec=baseline.decision_spec,
                current_spec=current_spec,
            )
            if reg_report.infra_config_mismatch:
                diff = ", ".join(
                    f"{key}: {b} -> {c}"
                    for key, (b, c) in sorted(reg_report.infra_config_diff.items())
                )
                print(
                    f"[tracelens] note: infra config mismatch vs baseline for "
                    f"task '{task_summary.task_id}' ({diff}); regressions "
                    f"within the {detector.noise_band_absolute} noise band "
                    f"are flagged but not blocking"
                )
            if reg_report.should_block_ci(threshold):
                print(reg_report.to_ci_output())
                blocking += 1

        # A gate that prints nothing on success is indistinguishable from
        # a gate that never ran — always say what was checked.
        unevaluable = checked == 0 or bool(no_gradable) or bool(no_comparable_metrics)
        summary = [f"{checked} checked", f"{len(skipped)} skipped (no baseline)"]
        if no_gradable:
            summary.append(f"{len(no_gradable)} skipped (no gradable trials)")
        if no_comparable_metrics:
            summary.append(f"{len(no_comparable_metrics)} skipped (no comparable metrics)")
        summary.append(f"{blocking} blocking regression(s)")
        if unevaluable:
            summary.append("UNEVALUABLE")
        print(f"[tracelens] Baseline check: {', '.join(summary)}")

        # Missing evidence invalidates the check even if another task regressed.
        if unevaluable:
            print(
                "Error: baseline check is unevaluable; verify the eval set, "
                "run count, and matching baseline metrics; fix any infra/grader "
                "failures listed above and rerun. This is not a passing gate.",
                file=sys.stderr,
            )
            return 2

        if skipped and args.require_baselines:
            print(
                f"Error: --require-baselines set but {len(skipped)} task(s) "
                f"have no baseline: {', '.join(skipped)}",
                file=sys.stderr,
            )
            return 1
        if blocking:
            return 1

    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Execute the 'report' subcommand."""
    with open(args.results) as f:
        data = json.load(f)

    report = ReportData.from_dict(data)
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
