"""`tracelens inspect` on a real saved run with every failure kind (issue #52)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tracelens.cli.inspect import cmd_inspect
from tracelens.cli.main import build_parser
from tracelens.core.grader import CodeGrader
from tracelens.core.task import EvalSet, Task, TaskExpectation
from tracelens.core.transcript import StepType, Transcript, TranscriptStep
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig


class _Agent(AgentAdapter):
    async def run(self, task: Task) -> Transcript:
        behaviour = task.input_data["behaviour"]
        if behaviour == "infra":
            raise ConnectionError("connection refused")
        if behaviour == "slow":
            await asyncio.sleep(0.5)
        transcript = Transcript(task_id=task.task_id)
        transcript.add_step(TranscriptStep(
            step_type=StepType.LLM_CALL, content="thinking " * 80, tokens_in=10, tokens_out=5,
        ))
        transcript.final_output = {"answer": "4" if behaviour == "pass" else "5"}
        return transcript


class _Grader(CodeGrader):
    def __init__(self) -> None:
        super().__init__("exact")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        if task.input_data["behaviour"] == "crash":
            raise ValueError("rubric missing")
        answer = transcript.final_output.get("answer")
        return {"correct": 1.0 if answer == task.metadata["expected"] else 0.0}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return metrics["correct"] == 1.0, metrics["correct"]


def _tasks() -> list[Task]:
    return [
        Task(task_id=f"t-{b}", name=f"{b} task", input_data={"behaviour": b},
             metadata={"expected": "4"}, expectation=TaskExpectation(expected_output="4"))
        for b in ("pass", "fail", "infra", "crash", "slow")
    ]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("inspect")
    runner = EvaluationRunner(_Agent(), [_Grader()], RunnerConfig(timeout_seconds=0.05))
    batch = asyncio.run(runner.run(EvalSet(name="suite", tasks=_tasks())))
    trials = root / "trials.json"
    trials.write_text(json.dumps(batch.to_dict()))
    eval_set = root / "tasks.json"
    eval_set.write_text(json.dumps({"tasks": [t.model_dump(mode="json") for t in _tasks()]}))
    return {"trials": trials, "eval_set": eval_set, "root": root}


def _run(*argv: str) -> int:
    return cmd_inspect(build_parser().parse_args(["inspect", *argv]))


def test_default_shows_every_failure_kind_and_no_passes(artifacts, capsys):
    assert _run(str(artifacts["trials"])) == 0
    out = capsys.readouterr().out
    assert "5 trial(s)" in out
    assert "passed 1, agent failure 2, infra error 1, grader error 1, not run 0" in out
    assert "Selected 4 trial(s)" in out
    assert "t-fail run 0  agent failure  status=completed" in out
    assert "t-slow run 0  agent failure  status=timeout" in out
    assert "t-infra run 0  infra error  status=infra_error" in out
    assert "error:    connection refused" in out
    assert "t-crash run 0  grader error" in out
    assert "exact CRASHED score=0.00" in out and "rubric missing" in out
    assert "t-pass" not in out
    assert 'actual:   {"answer": "5"}' in out
    assert "1 step(s) (1 shown), 15 tokens, 1 llm call(s)" in out
    assert "more characters" in out  # the long content is bounded


def test_filters_do_not_conflate_kinds(artifacts, capsys):
    assert _run(str(artifacts["trials"]), "--kind", "infra") == 0
    out = capsys.readouterr().out
    assert "Selected 1 trial(s) (kinds: infra error)" in out and "t-infra" in out
    assert _run(str(artifacts["trials"]), "--kind", "agent", "--task-id", "t-slow") == 0
    out = capsys.readouterr().out
    assert "Selected 1 trial(s)" in out and "t-slow" in out and "t-fail" not in out
    assert _run(str(artifacts["trials"]), "--grader", "exact") == 0
    out = capsys.readouterr().out
    assert "failed by grader: exact" in out and "t-fail" in out and "t-crash" in out
    assert "t-infra" not in out  # the grader never saw it
    assert _run(str(artifacts["trials"]), "--all") == 0
    assert "t-pass run 0  passed" in capsys.readouterr().out


def test_eval_set_adds_expected_name_and_input(artifacts, capsys):
    assert _run(str(artifacts["trials"]), "--task-id", "t-fail", "--eval-set", str(artifacts["eval_set"])) == 0
    out = capsys.readouterr().out
    assert "task:     fail task" in out
    assert 'input:    {"behaviour": "fail"}' in out
    assert "expected: 4" in out and 'actual:   {"answer": "5"}' in out
    assert "not supplied" not in out


def test_html_json_limit_and_full(artifacts, capsys):
    html = artifacts["root"] / "out" / "failures.html"
    data = artifacts["root"] / "out" / "failures.json"
    assert _run(str(artifacts["trials"]), "--limit", "1", "--html", str(html), "--json", str(data)) == 0
    captured = capsys.readouterr()
    assert "Selected 4 trial(s) (kinds: agent failure, infra error, grader error); showing the first 1" in captured.out
    assert f"[tracelens] wrote inspection html: {html}" in captured.err
    assert f"[tracelens] wrote inspection json: {data}" in captured.err
    page = html.read_text()
    assert page.startswith("<!DOCTYPE html>") and "t-crash" in page and "src=" not in page
    record = json.loads(data.read_text())
    assert record["selected"] == 4 and record["shown"] == 1 and record["trials"][0]["task_id"] == "t-crash"
    assert _run(str(artifacts["trials"]), "--task-id", "t-fail", "--full") == 0
    out = capsys.readouterr().out
    assert "more characters" not in out and "Unbounded output (--full)" in out


@pytest.mark.parametrize(
    ("argv", "fragment"),
    [
        (["missing.json"], "trials file not found"),
        (["TRIALS", "--max-chars", "-1"], "--max-chars cannot be negative"),
        (["TRIALS", "--limit", "0"], "--limit must be at least 1"),
        (["TRIALS", "--eval-set", "nope.json"], "not found"),
    ],
)
def test_input_errors_exit_2(artifacts, argv, fragment, capsys):
    argv = [str(artifacts["trials"]) if a == "TRIALS" else a for a in argv]
    assert _run(*argv) == 2
    captured = capsys.readouterr()
    assert fragment in captured.err and captured.out == ""


def test_results_file_is_rejected(artifacts, capsys):
    results = artifacts["root"] / "results.json"
    results.write_text(json.dumps({"total_trials": 1, "total_tasks": 1, "task_summaries": []}))
    assert _run(str(results)) == 2
    assert "is a results file" in capsys.readouterr().err


def test_real_process_prints_the_report_only(artifacts):
    result = subprocess.run(
        [sys.executable, "-m", "tracelens.cli.main", "inspect", str(artifacts["trials"]),
         "--kind", "grader"],
        cwd=Path(__file__).resolve().parents[3], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("Inspected ") and "t-crash" in result.stdout
    assert result.stderr == ""
