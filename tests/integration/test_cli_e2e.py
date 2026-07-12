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


# --- Gate integrity: the baseline check must fail loudly, never silently ---


def test_baseline_check_without_baselines_file_errors_before_running(
    tasks_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--baseline-check with no --baselines-file is a usage error (exit 2),
    caught before any eval time is spent — not a silent skip."""
    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baseline-check",
    )

    assert exit_code == 2
    assert EchoAdapter.run_count == 0
    assert "--baselines-file" in capsys.readouterr().err


def test_baseline_check_with_missing_baselines_file_errors_before_running(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope" / "baselines.json"

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baseline-check",
        "--baselines-file", str(missing),
    )

    assert exit_code == 2
    assert EchoAdapter.run_count == 0
    err = capsys.readouterr().err
    assert "baselines file not found" in err
    assert str(missing) in err


def test_baseline_check_warns_and_counts_tasks_without_baseline(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A task with no stored baseline is skipped with a visible warning and
    shows up in the gate summary — never a silent skip."""
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
        "--baseline-check",
        "--baselines-file", str(baselines),
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "t-fail" in captured.err
    assert "no baseline" in captured.err
    assert "1 checked" in captured.out
    assert "1 skipped (no baseline)" in captured.out


def test_require_baselines_fails_when_any_task_has_no_baseline(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
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
        "--baseline-check",
        "--baselines-file", str(baselines),
        "--require-baselines",
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "--require-baselines" in err
    assert "t-fail" in err


def test_gate_summary_printed_when_all_tasks_checked(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Even a fully-green gate says what it checked — a passing gate that
    prints nothing is indistinguishable from a gate that never ran."""
    baselines = tmp_path / "baselines.json"
    manager = BaselineManager(baselines)
    for task_id, score in (("t-pass", 0.9), ("t-fail", 0.1)):
        baseline = TaskBaseline(task_id=task_id)
        baseline.add_metric(
            metric_name="pass_rate",
            value=1.0 if score >= 0.5 else 0.0,
            std=0.05,
            sample_size=10,
        )
        baseline.add_metric(
            metric_name="mean_score", value=score, std=0.05, sample_size=10
        )
        manager.set_baseline(baseline)
    manager.save()

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baseline-check",
        "--baselines-file", str(baselines),
    )

    assert exit_code == 0
    assert "2 checked, 0 skipped (no baseline)" in capsys.readouterr().out


def test_baselines_file_without_baseline_check_warns(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--baselines-file alone does nothing today; that must be visible."""
    baselines = tmp_path / "baselines.json"
    baselines.write_text("{}")

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baselines-file", str(baselines),
    )

    assert exit_code == 0
    assert "no effect without --baseline-check" in capsys.readouterr().err
