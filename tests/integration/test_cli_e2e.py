"""End-to-end tests for the tracelens CLI.

Exercises the real user workflow — ``tracelens run`` with dotted-path
adapter/grader loading, output/report/trials files, baseline gating exit
codes, checkpoint resume, and ``tracelens report`` rendering — without
mocking any internals.
"""

import json
import sys
from pathlib import Path

import pytest

from tracelens.baselines.manager import BaselineManager, TaskBaseline
from tracelens.cli.main import build_parser, cmd_report, cmd_run, main
from tracelens.core.grader import CodeGrader
from tracelens.core.task import Task
from tracelens.core.transcript import StepType, Transcript, TranscriptStep
from tracelens.core.trial import TrialBatch
from tracelens.execution.agent_adapter import AgentAdapter

ADAPTER = "tests.integration.test_cli_e2e.EchoAdapter"
GRADER = "tests.integration.test_cli_e2e.ValueGrader"


class EchoAdapter(AgentAdapter):
    """Echoes task input back; loadable by dotted path from the CLI."""

    run_count = 0

    async def run(self, task: Task) -> Transcript:
        type(self).run_count += 1
        transcript = Transcript(task_id=task.task_id, final_output=dict(task.input_data))
        transcript.add_step(
            TranscriptStep(step_type=StepType.LLM_CALL, tokens_in=10, tokens_out=5)
        )
        return transcript


class ValueGrader(CodeGrader):
    """Passes when the echoed input value is >= 0.5."""

    def __init__(self) -> None:
        super().__init__("value_grader")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"value": float(transcript.final_output.get("value", 0.0))}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["value"] >= 0.5, metrics["value"]


@pytest.fixture(autouse=True)
def _reset_adapter_counter() -> None:
    EchoAdapter.run_count = 0


@pytest.fixture
def tasks_file(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({
        "tasks": [
            {
                "task_id": "t-pass",
                "name": "passing task",
                "description": "echoes a high value",
                "input_data": {"value": 0.9},
            },
            {
                "task_id": "t-fail",
                "name": "failing task",
                "description": "echoes a low value",
                "input_data": {"value": 0.1},
            },
        ]
    }))
    return path


def _run_cli(*argv: str) -> int:
    args = build_parser().parse_args(list(argv))
    return cmd_run(args)


def _run_main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["tracelens", *argv])
    with pytest.raises(SystemExit) as exc_info:
        main()
    code = exc_info.value.code
    return 0 if code is None else int(code)


def test_run_produces_output_report_and_trials(
    tasks_file: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.json"
    report_md = tmp_path / "report.md"
    trials = tmp_path / "trials.json"

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--num-runs", "2",
        "--output", str(out),
        "--report", str(report_md),
        "--save-trials", str(trials),
    )

    assert exit_code == 0

    data = json.loads(out.read_text())
    assert data["total_trials"] == 4
    assert data["total_tasks"] == 2
    assert data["overall_pass_rate"] == 0.5
    assert data["total_input_tokens"] == 40
    assert data["total_output_tokens"] == 20

    assert "t-pass" in report_md.read_text()

    batch = TrialBatch.from_dict(json.loads(trials.read_text()))
    assert batch.total_count == 4
    assert all(t.transcript is not None for t in batch.trials)


def test_report_command_renders_all_formats(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out.json"
    assert _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--output", str(out),
    ) == 0
    capsys.readouterr()  # discard run output

    for fmt in ("markdown", "json", "html"):
        args = build_parser().parse_args(
            ["report", "--results", str(out), "--format", fmt]
        )
        assert cmd_report(args) == 0
        rendered = capsys.readouterr().out
        assert "t-pass" in rendered


def test_baseline_check_blocks_on_regression(
    tasks_file: Path, tmp_path: Path
) -> None:
    baselines = tmp_path / "baselines.json"
    manager = BaselineManager(baselines)
    baseline = TaskBaseline(task_id="t-fail")
    baseline.add_metric(metric_name="pass_rate", value=1.0, std=0.05, sample_size=10)
    manager.set_baseline(baseline)
    manager.save()

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--num-runs", "2",
        "--baseline-check",
        "--baselines-file", str(baselines),
        "--fail-on-regression", "moderate",
    )

    assert exit_code == 1


def test_baseline_check_passes_without_regression(
    tasks_file: Path, tmp_path: Path
) -> None:
    baselines = tmp_path / "baselines.json"
    manager = BaselineManager(baselines)
    baseline = TaskBaseline(task_id="t-pass")
    baseline.add_metric(metric_name="pass_rate", value=1.0, std=0.05, sample_size=10)
    baseline.add_metric(metric_name="mean_score", value=0.9, std=0.05, sample_size=10)
    manager.set_baseline(baseline)
    manager.save()

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--num-runs", "2",
        "--baseline-check",
        "--baselines-file", str(baselines),
    )

    assert exit_code == 0


def test_checkpoint_resume_skips_completed_trials(
    tasks_file: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    argv = (
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--checkpoint", str(checkpoint),
    )

    assert _run_cli(*argv) == 0
    assert EchoAdapter.run_count == 2

    # Re-run with the same checkpoint: everything already done.
    assert _run_cli(*argv) == 0
    assert EchoAdapter.run_count == 2


def test_progress_flag_prints_to_stderr(
    tasks_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--progress",
    ) == 0

    err = capsys.readouterr().err
    assert "2/2 trials complete" in err


def test_init_scaffolds_a_runnable_eval_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "starter"
    project.mkdir()
    monkeypatch.chdir(project)

    assert _run_main(monkeypatch, "init", ".") == 0

    expected_files = {
        "eval/__init__.py",
        "eval/tasks.json",
        "eval/adapter.py",
        "eval/grader.py",
        "eval/README.md",
        ".github/workflows/eval.yml",
    }
    assert expected_files == {
        str(path.relative_to(project))
        for path in project.rglob("*")
        if path.is_file()
    }

    args = build_parser().parse_args([
        "run",
        "--eval-set", "eval/tasks.json",
        "--adapter", "eval.adapter.StarterAdapter",
        "--graders", "eval.grader.StarterGrader",
        "--output", "eval/results/results.json",
        "--report", "eval/results/report.md",
        "--save-trials", "eval/results/trials.json",
    ])

    assert cmd_run(args) == 0
    assert json.loads((project / "eval/results/results.json").read_text())[
        "overall_pass_rate"
    ] == 1.0


def test_init_refuses_to_overwrite_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "starter"
    project.mkdir()
    monkeypatch.chdir(project)

    assert _run_main(monkeypatch, "init", ".") == 0
    adapter = project / "eval/adapter.py"
    adapter.write_text("# user edit\n")

    assert _run_main(monkeypatch, "init", ".") == 1
    assert adapter.read_text() == "# user edit\n"

    assert _run_main(monkeypatch, "init", ".", "--force") == 0
    assert "class StarterAdapter" in adapter.read_text()
