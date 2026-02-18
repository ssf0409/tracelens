"""Tests for CLI argument parsing and defaults."""

import pytest

from eval_kit.cli.main import build_parser


class TestBuildParser:
    def test_run_required_args(self):
        """Run command requires eval-set, adapter, and graders."""
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--eval-set", "tasks.json",
            "--adapter", "my.Adapter",
            "--graders", "my.Grader1", "my.Grader2",
        ])
        assert args.command == "run"
        assert args.eval_set == "tasks.json"
        assert args.adapter == "my.Adapter"
        assert args.graders == ["my.Grader1", "my.Grader2"]

    def test_run_defaults(self):
        """Run command has sensible defaults."""
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--eval-set", "tasks.json",
            "--adapter", "my.Adapter",
            "--graders", "my.Grader",
        ])
        assert args.num_runs == 1
        assert args.max_concurrency == 5
        assert args.timeout == 300.0
        assert args.baseline_check is False
        assert args.fail_on_regression == "moderate"
        assert args.output is None
        assert args.report is None

    def test_run_with_all_options(self):
        """Run command accepts all optional args."""
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--eval-set", "tasks.json",
            "--adapter", "my.Adapter",
            "--graders", "my.Grader",
            "--num-runs", "5",
            "--max-concurrency", "10",
            "--timeout", "60",
            "--baseline-check",
            "--baselines-file", "baselines.json",
            "--fail-on-regression", "severe",
            "--output", "results.json",
            "--report", "report.md",
        ])
        assert args.num_runs == 5
        assert args.max_concurrency == 10
        assert args.timeout == 60.0
        assert args.baseline_check is True
        assert args.baselines_file == "baselines.json"
        assert args.fail_on_regression == "severe"
        assert args.output == "results.json"
        assert args.report == "report.md"

    def test_report_required_args(self):
        """Report command requires results file."""
        parser = build_parser()
        args = parser.parse_args([
            "report",
            "--results", "results.json",
        ])
        assert args.command == "report"
        assert args.results == "results.json"
        assert args.format == "markdown"

    def test_report_json_format(self):
        """Report command accepts JSON format."""
        parser = build_parser()
        args = parser.parse_args([
            "report",
            "--results", "results.json",
            "--format", "json",
        ])
        assert args.format == "json"

    def test_missing_command(self):
        """No command raises SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_missing_required_args(self):
        """Missing required args raises SystemExit."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])
