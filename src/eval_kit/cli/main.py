"""CLI entry point for eval-kit.

Usage:
    eval-kit run --eval-set tasks.json --adapter my.Adapter --graders my.Grader1 my.Grader2
    eval-kit report --results results.json --format markdown
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from eval_kit.baselines.comparison import RegressionDetector, RegressionSeverity
from eval_kit.baselines.manager import BaselineManager
from eval_kit.core.task import JSONTaskLoader, EvalSet
from eval_kit.execution.agent_adapter import AgentAdapter
from eval_kit.execution.registry import load_class
from eval_kit.execution.runner import EvaluationRunner, RunnerConfig
from eval_kit.reporting.generator import ReportData, ReportGenerator


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for eval-kit CLI."""
    parser = argparse.ArgumentParser(
        prog="eval-kit",
        description="Evaluation framework for AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- eval-kit run --
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
        help="Check results against baselines",
    )
    run_parser.add_argument(
        "--baselines-file", default=None,
        help="Path to baselines JSON file",
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

    # -- eval-kit report --
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

    return parser


def _severity_from_str(s: str) -> RegressionSeverity:
    return RegressionSeverity(s)


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    # Load eval set
    loader = JSONTaskLoader()
    tasks = loader.load(args.eval_set)
    eval_set = EvalSet(name=Path(args.eval_set).stem, tasks=tasks)

    # Load adapter and graders
    adapter_cls = load_class(args.adapter)
    adapter: AgentAdapter = adapter_cls()

    graders = []
    for grader_path in args.graders:
        grader_cls = load_class(grader_path)
        graders.append(grader_cls())

    # Build runner config
    config = RunnerConfig(
        num_runs=args.num_runs,
        max_concurrency=args.max_concurrency,
        timeout_seconds=args.timeout,
    )

    # Run evaluation
    runner = EvaluationRunner(adapter, graders, config)
    batch = asyncio.run(runner.run(eval_set))

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

    # CI summary to stdout
    print(gen.render_ci_summary(report))

    # Baseline check
    if args.baseline_check and args.baselines_file:
        manager = BaselineManager(args.baselines_file)
        detector = RegressionDetector()

        threshold = _severity_from_str(args.fail_on_regression)

        has_blocking = False
        for task_summary in report.task_summaries:
            baseline = manager.get_baseline(task_summary.task_id)
            if baseline:
                current_results = [
                    {"pass_rate": task_summary.pass_rate, "mean_score": task_summary.mean_score}
                ]
                reg_report = detector.compare(baseline, current_results)
                if reg_report.should_block_ci(threshold):
                    print(reg_report.to_ci_output())
                    has_blocking = True

        if has_blocking:
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
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
