"""End-to-end tests for the tracelens CLI.

Exercises the real user workflow — ``tracelens run`` with dotted-path
adapter/grader loading, output/report/trials files, baseline gating exit
codes, checkpoint resume, and ``tracelens report`` rendering — without
mocking any internals.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tracelens.baselines.manager import BaselineManager, TaskBaseline
from tracelens.cli.main import build_parser, cmd_report, cmd_run, main
from tracelens.core.decision_spec import DecisionSpec, InfraConfig
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
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9})

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


def test_corrupt_checkpoint_fails_cleanly(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A corrupt checkpoint is a misconfigured run: exit 2, not a traceback."""
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{not valid json")

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--checkpoint", str(checkpoint),
    )

    assert exit_code == 2
    assert "checkpoint" in capsys.readouterr().err.lower()
    assert EchoAdapter.run_count == 0


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


def _write_pass_baseline(tmp_path: Path, task_ids: dict[str, float]) -> Path:
    """Write a baselines file with pass_rate/mean_score metrics per task."""
    baselines = tmp_path / "baselines.json"
    manager = BaselineManager(baselines)
    for task_id, score in task_ids.items():
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
    return baselines


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
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9})

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
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9})

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
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9, "t-fail": 0.1})

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


class DiskFullAdapter(AgentAdapter):
    """Raises OSError; loadable by dotted path from the CLI."""

    async def run(self, task: Task) -> Transcript:
        raise OSError(28, "No space left on device")


def test_infra_exceptions_flag_extends_classification(
    tasks_file: Path, tmp_path: Path
) -> None:
    """--infra-exceptions lets CI declare which exception types are infra
    (downstream policy), so those failures land in infra_error_rate
    instead of masquerading as agent failures."""
    out = tmp_path / "out.json"

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", "tests.integration.test_cli_e2e.DiskFullAdapter",
        "--graders", GRADER,
        "--infra-exceptions", "builtins.OSError",
        "--output", str(out),
    )

    assert exit_code == 0
    assert json.loads(out.read_text())["infra_error_rate"] == 1.0


def test_infra_exceptions_flag_rejects_non_exception_types(
    tasks_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--infra-exceptions", "builtins.str",
    )

    assert exit_code == 2
    assert "not an exception type" in capsys.readouterr().err


def test_infra_exceptions_flag_rejects_unimportable_path(
    tasks_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--infra-exceptions", "no.such.Error",
    )

    assert exit_code == 2
    assert "no.such.Error" in capsys.readouterr().err


# --- Noise-aware gating from the CLI (DecisionSpec wiring) ---


class SpecStampingAdapter(AgentAdapter):
    """Echoes input and stamps a DecisionSpec on the transcript, the way
    a real adapter that knows its runtime config would."""

    async def run(self, task: Task) -> Transcript:
        transcript = Transcript(
            task_id=task.task_id, final_output=dict(task.input_data)
        )
        transcript.decision_spec = DecisionSpec(
            infra=InfraConfig(memory_hard_limit_mb=512)
        )
        return transcript


@pytest.fixture
def noise_tasks_file(tmp_path: Path) -> Path:
    """One task whose mean_score lands 2pp (absolute) below baseline —
    inside the 3pp infra-noise band but a 10% relative drop, so it
    blocks unless noise-awareness kicks in."""
    path = tmp_path / "noise_tasks.json"
    path.write_text(json.dumps({
        "tasks": [
            {
                "task_id": "t-noise",
                "name": "noise task",
                "description": "echoes a value 2pp under baseline",
                "input_data": {"value": 0.18},
            },
        ]
    }))
    return path


def _noise_baseline(
    tmp_path: Path, with_spec: bool
) -> Path:
    baselines = tmp_path / "baselines.json"
    manager = BaselineManager(baselines)
    baseline = TaskBaseline(
        task_id="t-noise",
        decision_spec=(
            DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=2048))
            if with_spec
            else None
        ),
    )
    baseline.add_metric(
        metric_name="mean_score", value=0.2, std=0.001, sample_size=10
    )
    manager.set_baseline(baseline)
    manager.save()
    return baselines


def _write_current_spec(tmp_path: Path) -> Path:
    spec_path = tmp_path / "current_spec.json"
    spec_path.write_text(json.dumps(
        DecisionSpec(
            infra=InfraConfig(memory_hard_limit_mb=512)
        ).model_dump(mode="json")
    ))
    return spec_path


def test_gate_blocks_small_delta_without_specs(
    noise_tasks_file: Path, tmp_path: Path
) -> None:
    """Control: with no DecisionSpec on either side, the 10%-relative
    regression blocks as before."""
    baselines = _noise_baseline(tmp_path, with_spec=False)

    exit_code = _run_cli(
        "run",
        "--eval-set", str(noise_tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baseline-check",
        "--baselines-file", str(baselines),
    )

    assert exit_code == 1


def test_decision_spec_file_enables_noise_aware_gate(
    noise_tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--decision-spec + a baseline that carries its spec: a sub-noise-band
    delta under mismatched infra no longer blocks, and the mismatch is
    called out in the output."""
    baselines = _noise_baseline(tmp_path, with_spec=True)
    spec_path = _write_current_spec(tmp_path)

    exit_code = _run_cli(
        "run",
        "--eval-set", str(noise_tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baseline-check",
        "--baselines-file", str(baselines),
        "--decision-spec", str(spec_path),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "infra config mismatch" in captured.out.lower()


def test_adapter_stamped_spec_enables_noise_aware_gate(
    noise_tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No flag needed when the adapter stamps DecisionSpecs on its
    transcripts — the gate picks the current spec up from the batch."""
    baselines = _noise_baseline(tmp_path, with_spec=True)

    exit_code = _run_cli(
        "run",
        "--eval-set", str(noise_tasks_file),
        "--adapter", "tests.integration.test_cli_e2e.SpecStampingAdapter",
        "--graders", GRADER,
        "--baseline-check",
        "--baselines-file", str(baselines),
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "infra config mismatch" in captured.out.lower()


def test_noise_band_flag_tightens_the_band(
    noise_tasks_file: Path, tmp_path: Path
) -> None:
    """--noise-band 0.01 shrinks the band below the 2pp delta, so the
    same mismatched-infra regression blocks again."""
    baselines = _noise_baseline(tmp_path, with_spec=True)
    spec_path = _write_current_spec(tmp_path)

    exit_code = _run_cli(
        "run",
        "--eval-set", str(noise_tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baseline-check",
        "--baselines-file", str(baselines),
        "--decision-spec", str(spec_path),
        "--noise-band", "0.01",
    )

    assert exit_code == 1


def test_decision_spec_flag_rejects_missing_file(
    noise_tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _run_cli(
        "run",
        "--eval-set", str(noise_tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--decision-spec", str(tmp_path / "nope.json"),
    )

    assert exit_code == 2
    assert "decision-spec" in capsys.readouterr().err


class FlakyInfraAdapter(AgentAdapter):
    """Raises ConnectionError; loadable by dotted path from the CLI."""

    async def run(self, task: Task) -> Transcript:
        raise ConnectionError("connection refused")


@pytest.mark.parametrize("require_baselines", [False, True])
@pytest.mark.parametrize(
    "adapter,grader",
    [
        ("tests.integration.test_cli_e2e.FlakyInfraAdapter", GRADER),
        (ADAPTER, "tests.integration.test_cli_e2e.CrashingGrader"),
    ],
)
def test_harness_failure_makes_baseline_gate_unevaluable(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    adapter: str, grader: str, require_baselines: bool,
) -> None:
    """Missing harness evidence is neither a regression nor a passing gate."""
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9, "t-fail": 0.1})
    output = tmp_path / "results.json"
    trials = tmp_path / "trials.json"

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", adapter,
        "--graders", grader,
        "--baseline-check",
        "--baselines-file", str(baselines),
        "--output", str(output),
        "--save-trials", str(trials),
        *(["--require-baselines"] if require_baselines else []),
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "no gradable trials" in captured.err
    assert "2 skipped (no gradable trials)" in captured.out
    assert "unevaluable" in captured.out.lower()
    assert "rerun" in captured.err.lower()
    assert "t-pass" in captured.err and "t-fail" in captured.err
    assert "REGRESSION DETECTED" not in captured.out
    assert json.loads(output.read_text())["total_trials"] == 2
    assert len(TrialBatch.from_dict(json.loads(trials.read_text())).trials) == 2


class CrashingGrader(ValueGrader):
    """A broken grading harness, not a negative agent observation."""

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        raise ValueError("invalid grading rubric")


class SomeTasksInfraAdapter(EchoAdapter):
    async def run(self, task: Task) -> Transcript:
        if task.task_id == "t-pass":
            raise ConnectionError("connection refused")
        return await super().run(task)


class SomeTasksCrashingGrader(ValueGrader):
    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        if task.task_id == "t-pass":
            raise ValueError("invalid grading rubric")
        return super().compute_metrics(transcript, task)


class PartialInfraAdapter(EchoAdapter):
    """Each task loses one trial but still has a valid comparison sample."""

    def __init__(self) -> None:
        self.seen: set[str] = set()

    async def run(self, task: Task) -> Transcript:
        if task.task_id not in self.seen:
            self.seen.add(task.task_id)
            raise ConnectionError("transient outage")
        return await super().run(task)


class SlowAdapter(EchoAdapter):
    async def run(self, task: Task) -> Transcript:
        await asyncio.sleep(60)
        return await super().run(task)


@pytest.mark.parametrize(
    "adapter",
    ["tests.integration.test_cli_e2e.DiskFullAdapter", "tests.integration.test_cli_e2e.SlowAdapter"],
)
def test_agent_execution_failures_remain_regression_observations(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], adapter: str
) -> None:
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9, "t-fail": 0.9})

    assert _run_cli(
        "run", "--eval-set", str(tasks_file), "--adapter", adapter,
        "--graders", GRADER, "--timeout", "0.01",
        "--baseline-check", "--baselines-file", str(baselines),
    ) == 1

    captured = capsys.readouterr()
    assert "2 checked" in captured.out
    assert "REGRESSION DETECTED" in captured.out
    assert "unevaluable" not in captured.out.lower()


@pytest.mark.parametrize(
    "adapter,grader",
    [
        ("tests.integration.test_cli_e2e.FlakyInfraAdapter", GRADER),
        (ADAPTER, "tests.integration.test_cli_e2e.CrashingGrader"),
    ],
)
def test_non_gated_harness_failures_keep_observational_exit_behavior(
    tasks_file: Path, capsys: pytest.CaptureFixture[str], adapter: str, grader: str
) -> None:
    assert _run_cli(
        "run", "--eval-set", str(tasks_file), "--adapter", adapter,
        "--graders", grader,
    ) == 0
    assert "Baseline check:" not in capsys.readouterr().out


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize(
    "adapter,grader",
    [
        ("tests.integration.test_cli_e2e.SomeTasksInfraAdapter", GRADER),
        (ADAPTER, "tests.integration.test_cli_e2e.SomeTasksCrashingGrader"),
    ],
)
def test_other_task_results_cannot_hide_an_unevaluable_task(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    adapter: str, grader: str, regression: bool,
) -> None:
    baselines = _write_pass_baseline(
        tmp_path, {"t-pass": 0.9, "t-fail": 0.9 if regression else 0.1}
    )

    assert _run_cli(
        "run", "--eval-set", str(tasks_file),
        "--adapter", adapter, "--graders", grader,
        "--baseline-check", "--baselines-file", str(baselines),
    ) == 2

    captured = capsys.readouterr()
    assert "1 checked" in captured.out
    assert "1 skipped (no gradable trials)" in captured.out
    assert "unevaluable" in captured.out.lower()
    assert ("REGRESSION DETECTED" in captured.out) == regression


@pytest.mark.parametrize("retry", [False, True])
def test_partial_trial_loss_keeps_gradable_task_comparisons(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], retry: bool
) -> None:
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9, "t-fail": 0.1})

    assert _run_cli(
        "run", "--eval-set", str(tasks_file),
        "--adapter", "tests.integration.test_cli_e2e.PartialInfraAdapter",
        "--graders", GRADER, "--num-runs", "1" if retry else "2",
        "--max-infra-retries", "1" if retry else "0",
        "--baseline-check", "--baselines-file", str(baselines),
        "--require-baselines",
    ) == 0

    captured = capsys.readouterr()
    assert "2 checked" in captured.out
    assert ("excluded 1" in captured.err) == (not retry)
    assert "REGRESSION DETECTED" not in captured.out


@pytest.mark.parametrize("case", ["empty-suite", "zero-runs", "empty-baselines", "unrelated-baselines"])
def test_baseline_gate_requires_at_least_one_comparison(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    baseline_values = {"t-pass": 0.9, "t-fail": 0.1}
    if case == "empty-suite":
        tasks_file.write_text('{"tasks": []}')
    elif case == "empty-baselines":
        baseline_values = {}
    elif case == "unrelated-baselines":
        baseline_values = {"foreign-task": 0.9}
    baselines = _write_pass_baseline(tmp_path, baseline_values)

    assert _run_cli(
        "run", "--eval-set", str(tasks_file),
        "--adapter", ADAPTER, "--graders", GRADER,
        "--num-runs", "0" if case == "zero-runs" else "1",
        "--baseline-check", "--baselines-file", str(baselines),
    ) == 2

    captured = capsys.readouterr()
    assert "0 checked" in captured.out
    assert "unevaluable" in captured.out.lower()
    assert "rerun" in captured.err.lower()


@pytest.mark.parametrize("metric", [None, "domain_quality"])
def test_task_without_comparable_baseline_metrics_invalidates_gate(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    metric: str | None,
) -> None:
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9})
    manager = BaselineManager(baselines)
    baseline = TaskBaseline(task_id="t-fail")
    if metric is not None:
        baseline.add_metric(metric, 0.9)
    manager.set_baseline(baseline)
    manager.save()

    assert _run_cli(
        "run", "--eval-set", str(tasks_file),
        "--adapter", ADAPTER, "--graders", GRADER,
        "--baseline-check", "--baselines-file", str(baselines),
    ) == 2

    captured = capsys.readouterr()
    assert "1 checked" in captured.out
    assert "no comparable metrics" in captured.out
    assert "t-fail" in captured.err
    assert "pass_rate" in captured.err and "mean_score" in captured.err


@pytest.mark.parametrize(
    "adapter,grader,expected",
    [
        (ADAPTER, GRADER, 0),
        (ADAPTER, GRADER, 1),
        ("tests.integration.test_cli_e2e.FlakyInfraAdapter", GRADER, 2),
        (ADAPTER, "tests.integration.test_cli_e2e.CrashingGrader", 2),
    ],
)
def test_baseline_exit_contract_in_real_cli_process(
    tasks_file: Path, tmp_path: Path, adapter: str, grader: str, expected: int
) -> None:
    baselines = _write_pass_baseline(
        tmp_path, {"t-pass": 0.9, "t-fail": 0.9 if expected == 1 else 0.1}
    )
    result = subprocess.run(
        [sys.executable, "-m", "tracelens.cli.main", "run",
         "--eval-set", str(tasks_file), "--adapter", adapter, "--graders", grader,
         "--baseline-check", "--baselines-file", str(baselines)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, timeout=30,
    )

    assert result.returncode == expected, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "Baseline check:" in result.stdout


def test_corrupt_baselines_file_errors_before_running(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baselines = tmp_path / "baselines.json"
    baselines.write_text("{not valid json")

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--baseline-check",
        "--baselines-file", str(baselines),
    )

    assert exit_code == 2
    assert EchoAdapter.run_count == 0
    assert "could not load baselines file" in capsys.readouterr().err


def test_require_baselines_without_baseline_check_errors(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9})

    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--require-baselines",
        "--baselines-file", str(baselines),
    )

    assert exit_code == 2
    assert EchoAdapter.run_count == 0
    assert "--baseline-check" in capsys.readouterr().err


def test_noise_band_without_baseline_check_errors(
    tasks_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _run_cli(
        "run",
        "--eval-set", str(tasks_file),
        "--adapter", ADAPTER,
        "--graders", GRADER,
        "--noise-band", "0.05",
    )

    assert exit_code == 2
    assert "--baseline-check" in capsys.readouterr().err


# --- Issue #47: the gate decision is persisted and re-rendered -------------


def test_gate_decision_is_persisted_and_rerendered(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A blocked gate is recorded in JSON/Markdown and survives `tracelens report`."""
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9, "t-fail": 0.9})
    output = tmp_path / "results.json"
    report_md = tmp_path / "report.md"
    report_html = tmp_path / "report.html"

    exit_code = _run_cli(
        "run", "--eval-set", str(tasks_file), "--adapter", ADAPTER, "--graders", GRADER,
        "--num-runs", "2", "--baseline-check", "--baselines-file", str(baselines),
        "--fail-on-regression", "moderate",
        "--output", str(output), "--report", str(report_md), "--html-report", str(report_html),
    )
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "REGRESSION DETECTED" in captured.out
    assert "Baseline check:" in captured.out
    assert "blocking regression" in captured.err

    data = json.loads(output.read_text())
    gate = data["gate"]
    assert gate["status"] == "blocked" and gate["exit_code"] == 1
    assert gate["threshold"] == "moderate" and gate["checked"] == 2
    failing = next(t for t in gate["tasks"] if t["task_id"] == "t-fail")
    assert failing["outcome"] == "checked" and failing["blocking"]
    assert failing["regressions"][0]["metric_name"] == "pass_rate"
    assert failing["regressions"][0]["severity"] == "severe"

    for text in (report_md.read_text(), report_html.read_text()):
        assert "Baseline Gate" in text and "BLOCKED" in text
        assert "t-fail" in text and "pass_rate" in text and "severe" in text

    # Re-render from the saved JSON: identical decision, no recomputation.
    assert _run_report(capsys, output, "markdown") == 0
    rendered = capsys.readouterr().out
    assert "**Status**: BLOCKED (exit code 1)" in rendered
    assert "block at `moderate` or worse" in rendered
    assert "| t-fail | pass_rate |" in rendered and "severe" in rendered
    assert _run_report(capsys, output, "json") == 0
    assert json.loads(capsys.readouterr().out)["gate"] == gate


def _run_report(capsys: pytest.CaptureFixture[str], results: Path, fmt: str) -> int:
    capsys.readouterr()  # drop anything pending
    args = build_parser().parse_args(["report", "--results", str(results), "--format", fmt])
    return cmd_report(args)


def test_unevaluable_and_non_gated_decisions_are_persisted(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baselines = _write_pass_baseline(tmp_path, {"t-pass": 0.9, "t-fail": 0.1})
    output = tmp_path / "results.json"

    assert _run_cli(
        "run", "--eval-set", str(tasks_file),
        "--adapter", "tests.integration.test_cli_e2e.FlakyInfraAdapter", "--graders", GRADER,
        "--baseline-check", "--baselines-file", str(baselines), "--output", str(output),
    ) == 2
    capsys.readouterr()
    gate = json.loads(output.read_text())["gate"]
    assert gate["status"] == "unevaluable" and gate["exit_code"] == 2
    assert gate["skipped_no_gradable"] == 2
    assert all(t["outcome"] == "no_gradable_trials" for t in gate["tasks"])
    # All-infra stays distinguishable from an agent regression in the same document.
    assert json.loads(output.read_text())["infra_error_count"] == 2
    assert gate["blocking_regressions"] == 0

    assert _run_cli(
        "run", "--eval-set", str(tasks_file), "--adapter", ADAPTER, "--graders", GRADER,
        "--output", str(output),
    ) == 0
    capsys.readouterr()
    assert json.loads(output.read_text())["gate"] == {
        **json.loads(output.read_text())["gate"], "status": "not_requested", "exit_code": 0,
    }


def test_report_rejects_files_that_are_not_results(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps({"foo": 1}))
    assert _run_report(capsys, unrelated, "markdown") == 2
    assert "not a TraceLens results file" in capsys.readouterr().err

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert _run_report(capsys, broken, "markdown") == 2
    assert "invalid JSON" in capsys.readouterr().err

    assert _run_report(capsys, tmp_path / "missing.json", "markdown") == 2
    assert "not found" in capsys.readouterr().err


def test_output_write_failure_is_a_clear_error(
    tasks_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file.txt"
    blocker.write_text("not a directory")
    exit_code = _run_cli(
        "run", "--eval-set", str(tasks_file), "--adapter", ADAPTER, "--graders", GRADER,
        "--output", str(blocker / "results.json"),
    )
    assert exit_code == 2
    captured = capsys.readouterr()
    assert "could not write output file" in captured.err
    assert "Traceback" not in captured.err


# --- Issue #49: the generated scaffold's gate walkthrough actually works ----


def _forget_scaffold_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop cached ``eval.*`` modules so a rewritten adapter is re-imported."""
    for name in list(sys.modules):
        if name == "eval" or name.startswith("eval."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_init_gate_walkthrough_blocks_an_intentional_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Follow eval/README.md step 4 literally: store baselines with the README's
    own snippet, then prove a broken agent is blocked and a fixed one passes."""
    import textwrap

    import yaml

    project = tmp_path / "starter"
    project.mkdir()
    monkeypatch.chdir(project)
    _forget_scaffold_modules(monkeypatch)
    assert _run_main(monkeypatch, "init", ".") == 0

    # The generated workflow is valid YAML that evaluates every pull request.
    workflow = yaml.safe_load((project / ".github/workflows/eval.yml").read_text())
    triggers = workflow.get("on", workflow.get(True))
    assert triggers["pull_request"] == {"branches": ["main"]}

    run = [
        "run", "--eval-set", "eval/tasks.json",
        "--adapter", "eval.adapter.StarterAdapter", "--graders", "eval.grader.StarterGrader",
        "--output", "eval/results/results.json", "--report", "eval/results/report.md",
    ]
    gate = run + [
        "--baseline-check", "--baselines-file", "eval/baselines.json",
        "--fail-on-regression", "moderate",
    ]
    results = project / "eval/results/results.json"

    # Step 1: the smoke run passes by construction.
    assert cmd_run(build_parser().parse_args(run)) == 0
    assert json.loads(results.read_text())["gate"]["status"] == "not_requested"

    # Step 4.1: store baselines with the snippet from the README, verbatim.
    readme = (project / "eval/README.md").read_text()
    snippet = readme.split("python - <<'EOF'\n", 1)[1].split("\n   EOF", 1)[0]
    exec(compile(textwrap.dedent(snippet), "eval/README.md", "exec"), {})
    assert (project / "eval/baselines.json").exists()

    # Step 4.2/4.3: the trusted agent passes the gate ...
    assert cmd_run(build_parser().parse_args(gate)) == 0
    assert json.loads(results.read_text())["gate"]["status"] == "passed"

    # ... an intentionally broken agent is blocked ...
    adapter = project / "eval/adapter.py"
    broken = adapter.read_text().replace(
        'return {"answer": input_data["answer"]}', 'return {"answer": "wrong"}'
    )
    assert broken != adapter.read_text()
    adapter.write_text(broken)
    _forget_scaffold_modules(monkeypatch)
    assert cmd_run(build_parser().parse_args(gate)) == 1
    data = json.loads(results.read_text())
    assert data["gate"]["status"] == "blocked" and data["gate"]["exit_code"] == 1
    assert data["gate"]["blocking_regressions"] == 2
    assert "BLOCKED" in (project / "eval/results/report.md").read_text()
    assert "REGRESSION DETECTED" in capsys.readouterr().out

    # ... and reverting the change passes again.
    _run_main(monkeypatch, "init", ".", "--force")
    _forget_scaffold_modules(monkeypatch)
    assert cmd_run(build_parser().parse_args(gate)) == 0


# --- Issue #50: JSONL and CSV eval sets through the CLI ----------------------


def _equivalent_task_files(tmp_path: Path) -> dict[str, Path]:
    """The tasks_file suite (t-pass 0.9 / t-fail 0.1) in JSONL and CSV."""
    import csv

    records = [
        {"task_id": "t-pass", "name": "passing task", "input": {"value": 0.9}},
        {"task_id": "t-fail", "name": "failing task", "input": {"value": 0.1}},
    ]
    jsonl = tmp_path / "tasks.jsonl"
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    csv_path = tmp_path / "tasks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["task_id", "name", "input"])
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "input": json.dumps(record["input"])})
    return {"jsonl": jsonl, "csv": csv_path}


@pytest.mark.parametrize("fmt", ["jsonl", "csv"])
def test_run_accepts_jsonl_and_csv_eval_sets(
    tasks_file: Path, tmp_path: Path, fmt: str, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_output = tmp_path / "json-results.json"
    assert _run_cli(
        "run", "--eval-set", str(tasks_file), "--adapter", ADAPTER, "--graders", GRADER,
        "--output", str(baseline_output),
    ) == 0
    other_output = tmp_path / f"{fmt}-results.json"
    assert _run_cli(
        "run", "--eval-set", str(_equivalent_task_files(tmp_path)[fmt]),
        "--adapter", ADAPTER, "--graders", GRADER, "--output", str(other_output),
    ) == 0
    capsys.readouterr()

    def _per_task(path: Path) -> dict[str, float]:
        data = json.loads(path.read_text())
        return {s["task_id"]: s["pass_rate"] for s in data["task_summaries"]}

    assert _per_task(other_output) == _per_task(baseline_output) == {"t-pass": 1.0, "t-fail": 0.0}


def test_run_maps_foreign_input_columns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        json.dumps({"task_id": "t-pass", "prompt": {"value": 0.9}, "subject": "geo"}) + "\n"
    )
    output = tmp_path / "results.json"
    assert _run_cli(
        "run", "--eval-set", str(prompts), "--input-field", "prompt",
        "--metadata-fields", "subject",
        "--adapter", ADAPTER, "--graders", GRADER, "--output", str(output),
    ) == 0
    capsys.readouterr()
    assert json.loads(output.read_text())["task_summaries"][0]["pass_rate"] == 1.0


@pytest.mark.parametrize(
    "setup, expected_error",
    [
        ("unknown-suffix", "unsupported eval-set file type '.yaml'"),
        ("directory-without-format", "--eval-set-format"),
        ("malformed-jsonl", "tasks.jsonl:2"),
        ("missing", "not found"),
    ],
)
def test_eval_set_load_failures_exit_2_before_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], setup: str, expected_error: str
) -> None:
    if setup == "unknown-suffix":
        target = tmp_path / "tasks.yaml"
        target.write_text("tasks: []\n")
    elif setup == "directory-without-format":
        target = tmp_path / "suite"
        target.mkdir()
    elif setup == "malformed-jsonl":
        target = tmp_path / "tasks.jsonl"
        target.write_text('{"input": {"value": 0.9}}\nnot json\n')
    else:
        target = tmp_path / "missing.json"

    assert _run_cli(
        "run", "--eval-set", str(target), "--adapter", ADAPTER, "--graders", GRADER,
    ) == 2
    captured = capsys.readouterr()
    assert expected_error in captured.err
    assert "Traceback" not in captured.err
    assert EchoAdapter.run_count == 0  # nothing ran
