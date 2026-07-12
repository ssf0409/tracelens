"""Tests for CLI argument parsing and defaults."""

import json
from pathlib import Path

import pytest

from tracelens.cli.calibrate import cmd_calibrate
from tracelens.cli.main import build_parser


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
        assert args.max_infra_retries == 0

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
            "--max-infra-retries", "2",
        ])
        assert args.num_runs == 5
        assert args.max_concurrency == 10
        assert args.timeout == 60.0
        assert args.baseline_check is True
        assert args.baselines_file == "baselines.json"
        assert args.fail_on_regression == "severe"
        assert args.output == "results.json"
        assert args.report == "report.md"
        assert args.max_infra_retries == 2

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


class TestCalibrateParser:
    def test_calibrate_required_args(self):
        """Calibrate command requires grader, samples, and annotations."""
        parser = build_parser()
        args = parser.parse_args([
            "calibrate",
            "--grader", "my.graders.QualityGrader",
            "--samples", "samples.json",
            "--annotations", "human_grades.json",
        ])
        assert args.command == "calibrate"
        assert args.grader == "my.graders.QualityGrader"
        assert args.samples == "samples.json"
        assert args.annotations == "human_grades.json"

    def test_calibrate_defaults(self):
        """Calibrate command has sensible defaults."""
        parser = build_parser()
        args = parser.parse_args([
            "calibrate",
            "--grader", "my.Grader",
            "--samples", "s.json",
            "--annotations", "a.json",
        ])
        assert args.threshold == 0.7
        assert args.output is None
        assert args.transcripts is None
        assert args.results is None

    def test_calibrate_with_all_options(self):
        """Calibrate command accepts all optional args."""
        parser = build_parser()
        args = parser.parse_args([
            "calibrate",
            "--grader", "my.Grader",
            "--samples", "s.json",
            "--annotations", "a.json",
            "--transcripts", "transcripts.json",
            "--threshold", "0.8",
            "--output", "cal.json",
        ])
        assert args.threshold == 0.8
        assert args.transcripts == "transcripts.json"
        assert args.output == "cal.json"


class TestCmdCalibrateIntegration:
    """Integration tests for the calibrate CLI command with temp files."""

    def _write_json(self, path: Path, data: object) -> None:
        with open(path, "w") as f:
            json.dump(data, f)

    def test_calibrate_with_precomputed_results(self, tmp_path: Path) -> None:
        """cmd_calibrate loads annotations + results and produces output."""
        annotations = [
            {"task_id": "t1", "human_score": 0.9, "human_passed": True},
            {"task_id": "t2", "human_score": 0.3, "human_passed": False},
            {"task_id": "t3", "human_score": 0.7, "human_passed": True},
        ]
        results = {
            "t1": {"trial_id": "x", "grader_id": "g", "passed": True, "score": 0.85},
            "t2": {"trial_id": "x", "grader_id": "g", "passed": False, "score": 0.25},
            "t3": {"trial_id": "x", "grader_id": "g", "passed": True, "score": 0.75},
        }
        ann_path = tmp_path / "annotations.json"
        res_path = tmp_path / "results.json"
        out_path = tmp_path / "calibration.json"
        samples_path = tmp_path / "samples.json"

        self._write_json(ann_path, annotations)
        self._write_json(res_path, results)
        self._write_json(samples_path, [])

        parser = build_parser()
        args = parser.parse_args([
            "calibrate",
            "--grader", "tracelens.core.grader.Grader",
            "--samples", str(samples_path),
            "--annotations", str(ann_path),
            "--results", str(res_path),
            "--output", str(out_path),
        ])
        exit_code = cmd_calibrate(args)

        assert out_path.exists()
        with open(out_path) as f:
            cal_data = json.load(f)
        assert cal_data["sample_count"] == 3
        assert cal_data["pearson_r"] is not None
        assert "pairs" in cal_data
        assert len(cal_data["pairs"]) == 3
        # High correlation expected — scores track well
        assert exit_code == 0

    def test_calibrate_no_transcripts_or_results_returns_error(self, tmp_path: Path) -> None:
        """cmd_calibrate returns 1 when neither --transcripts nor --results given."""
        ann_path = tmp_path / "annotations.json"
        samples_path = tmp_path / "samples.json"
        self._write_json(ann_path, [{"task_id": "t1", "human_score": 0.5, "human_passed": True}])
        self._write_json(samples_path, [])

        parser = build_parser()
        args = parser.parse_args([
            "calibrate",
            "--grader", "tracelens.core.grader.Grader",
            "--samples", str(samples_path),
            "--annotations", str(ann_path),
        ])
        assert cmd_calibrate(args) == 1
