"""Stage A of issue #33: the documented user journey, end to end.

Every step below is a real ``tracelens`` process run inside a scratch
project, exercising only documented commands and checking both the exit
code and the decision persisted in the artifacts. The executable is the
console script named by ``TRACELENS_CLI`` (CI points it at a wheel
installed into a clean environment); locally it falls back to
``python -m tracelens.cli.main`` from the current interpreter.

The journey: an existing project -> ``tracelens init`` -> ``run --config``
-> baselines from the README snippet -> gate enabled in ``tracelens.yaml``
-> an intentional regression blocks -> ``inspect`` explains it ->
``compare`` calls it a regression -> a targeted ``--task-id`` rerun ->
an infra outage and a grader crash make the gate unevaluable and are told
apart -> malformed input and a bad config are usage errors ->
checkpoint/resume re-executes nothing -> the suite passes again and
``compare`` calls it equivalent -> ``report`` and ``sample`` read the
artifacts.

This deliberately duplicates no unit test: the boundary it covers is the
installed console script driving the whole documented workflow on disk.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

CLI = (
    shlex.split(os.environ["TRACELENS_CLI"])
    if os.environ.get("TRACELENS_CLI")
    else [sys.executable, "-m", "tracelens.cli.main"]
)


def tracelens(*args: str, cwd: Path, expect: int) -> subprocess.CompletedProcess[str]:
    """Run one documented command and assert its exit code, with context on failure."""
    result = subprocess.run(
        [*CLI, *args], cwd=cwd, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == expect, (
        f"tracelens {' '.join(args)}\nexit {result.returncode}, expected {expect}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    assert "Traceback" not in result.stderr, result.stderr
    return result


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def enable_gate(config: Path) -> None:
    """What eval/README.md step 4.2 says: uncomment the ``baseline:`` block."""
    lines = config.read_text().splitlines(keepends=True)
    start = lines.index("  # baseline:\n")
    lines[start] = "  baseline:\n"
    for i in range(start + 1, len(lines)):
        if not lines[i].startswith("  #   "):
            break
        lines[i] = "    " + lines[i][len("  #   "):]
    config.write_text("".join(lines))


def readme_snippet(readme: Path) -> str:
    """The baseline-storing snippet from eval/README.md step 4.1, verbatim."""
    text = readme.read_text()
    return textwrap.dedent(text.split("python - <<'EOF'\n", 1)[1].split("\n   EOF", 1)[0])


TRUSTED = 'return {"answer": input_data["answer"]}'


def test_documented_user_journey(tmp_path: Path) -> None:
    project = tmp_path / "agent-project"
    project.mkdir()
    # An existing Python project: TraceLens must fit in next to what is there.
    (project / "pyproject.toml").write_text(
        '[project]\nname = "agent-project"\nversion = "0.1.0"\n'
    )
    results = project / "eval/results/results.json"
    trials = project / "eval/results/trials.json"
    report = project / "eval/results/report.md"
    config = project / "tracelens.yaml"
    adapter = project / "eval/adapter.py"
    grader = project / "eval/grader.py"
    tasks = project / "eval/tasks.json"

    # 1. Scaffold, and refuse to clobber it.
    init = tracelens("init", ".", cwd=project, expect=0)
    assert "Next: tracelens run --config tracelens.yaml" in init.stdout
    for relative in ("tracelens.yaml", "eval/tasks.json", "eval/adapter.py", "eval/grader.py",
                     "eval/README.md", ".github/workflows/eval.yml"):
        assert (project / relative).is_file(), relative
    assert (project / "pyproject.toml").read_text().startswith("[project]")
    tracelens("init", ".", cwd=project, expect=2)

    # 2. The one documented command: outputs land where the config says.
    run = tracelens("run", "--config", "tracelens.yaml", cwd=project, expect=0)
    assert run.stdout.startswith("TraceLens: 2 tasks, 2 trials, pass_rate=100.0%")
    assert f"[tracelens] wrote results: {results}" in run.stderr
    assert trials.is_file() and report.is_file() and (project / "eval/results/report.html").is_file()
    data = load(results)
    assert data["gate"]["status"] == "not_requested"
    assert set(data["provenance"]["measurement"]["task_hashes"]) == {"starter-capital", "starter-math"}

    # 3. Store baselines exactly as the README says.
    snippet = subprocess.run(
        [sys.executable, "-"], input=readme_snippet(project / "eval/README.md"),
        cwd=project, capture_output=True, text=True, timeout=120,
    )
    assert snippet.returncode == 0, snippet.stderr
    baselines = load(project / "eval/baselines.json")
    assert all(entry["task_hash"] for entry in baselines.values())

    # 4. Enable the gate in tracelens.yaml; the trusted agent passes it.
    enable_gate(config)
    run = tracelens("run", "--config", "tracelens.yaml", cwd=project, expect=0)
    assert "Baseline check: 2 checked, 0 skipped (no baseline), 0 blocking regression(s)" in run.stdout
    assert load(results)["gate"]["status"] == "passed"
    trusted_trials = project / "eval/results/trusted-trials.json"
    trusted_trials.write_text(trials.read_text())

    # 5. An intentional regression is blocked, and the decision is persisted.
    source = adapter.read_text()
    assert TRUSTED in source
    adapter.write_text(source.replace(TRUSTED, 'return {"answer": "wrong"}'))
    run = tracelens("run", "--config", "tracelens.yaml", cwd=project, expect=1)
    assert "REGRESSION DETECTED" in run.stdout
    gate = load(results)["gate"]
    assert gate["status"] == "blocked" and gate["exit_code"] == 1
    assert gate["blocking_regressions"] == 2
    assert "BLOCKED" in report.read_text()

    # 6. inspect explains the failure from the trials file.
    inspect = tracelens(
        "inspect", "eval/results/trials.json", "--failures", "--eval-set", "eval/tasks.json",
        "--html", "eval/results/failures.html", cwd=project, expect=0,
    )
    assert "passed 0, agent failure 2, infra error 0, grader error 0, not run 0" in inspect.stdout
    assert "starter-capital run 0  agent failure  status=completed" in inspect.stdout
    assert 'actual:   {"answer": "wrong"}' in inspect.stdout
    assert "task:     Answer a simple geography question" in inspect.stdout
    assert "starter FAIL score=0.00" in inspect.stdout
    assert (project / "eval/results/failures.html").read_text().count("agent failure") >= 2

    # 7. compare calls the broken run a regression against the trusted one.
    compare = tracelens(
        "compare", "eval/results/trusted-trials.json", "eval/results/trials.json",
        "--output", "eval/results/compare.json", cwd=project, expect=1,
    )
    assert "Verdict: REGRESSION (exit 1)" in compare.stdout
    assert "What changed: nothing declared" in compare.stdout  # same class path and spec
    decision = load(project / "eval/results/compare.json")
    assert decision["verdict"] == "regression" and decision["delta"] == -1.0
    assert decision["alignment"]["compared"] == 2 and decision["alignment"]["aligned_by"] == "content"

    # 8. Fix it and rerun only the affected task; the gate checks just that task.
    adapter.write_text(source)
    run = tracelens(
        "run", "--config", "tracelens.yaml", "--task-id", "starter-capital", cwd=project, expect=0,
    )
    assert "[tracelens] running 1 of 2 task(s): starter-capital" in run.stderr
    data = load(results)
    assert data["total_tasks"] == 1 and data["gate"]["status"] == "passed" and data["gate"]["checked"] == 1
    tracelens(
        "run", "--config", "tracelens.yaml", "--task-id", "no-such-task", cwd=project, expect=2,
    )

    # 9. An infra outage on one task is unevaluable, not a failure of the agent.
    adapter.write_text(source.replace(
        TRUSTED,
        'if input_data["question"].startswith("What is 2"):\n'
        '        raise ConnectionError("connection refused")\n'
        f"    {TRUSTED}",
    ))
    run = tracelens("run", "--config", "tracelens.yaml", cwd=project, expect=2)
    assert "UNEVALUABLE" in run.stdout
    gate = load(results)["gate"]
    assert gate["status"] == "unevaluable" and gate["skipped_no_gradable"] == 1
    inspect = tracelens("inspect", "eval/results/trials.json", "--kind", "infra", cwd=project, expect=0)
    assert "starter-math run 0  infra error  status=infra_error" in inspect.stdout
    assert "error:    connection refused" in inspect.stdout
    assert "starter-capital" not in inspect.stdout
    adapter.write_text(source)

    # 10. A grader crash is unevaluable too, and never counted against the agent.
    grader_source = grader.read_text()
    anchor = "    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:\n"
    assert anchor in grader_source
    grader.write_text(grader_source.replace(
        anchor, anchor + '        raise ValueError("rubric missing")\n'
    ))
    run = tracelens("run", "--config", "tracelens.yaml", cwd=project, expect=2)
    assert load(results)["gate"]["status"] == "unevaluable"
    assert load(results)["grader_error_count"] == 2
    inspect = tracelens("inspect", "eval/results/trials.json", "--kind", "grader", cwd=project, expect=0)
    assert "grader error 2" in inspect.stdout and "starter CRASHED score=0.00" in inspect.stdout
    assert "rubric missing" in inspect.stdout
    grader.write_text(grader_source)

    # 11. Malformed input and a bad config are usage errors, before any agent runs.
    task_source = tasks.read_text()
    tasks.write_text("{not json")
    run = tracelens("run", "--config", "tracelens.yaml", cwd=project, expect=2)
    assert "tasks.json" in run.stderr and run.stdout == ""
    tasks.write_text(task_source)
    bad_config = project / "bad.yaml"
    bad_config.write_text(config.read_text().replace("  num_runs: 1\n", "  num_run: 1\n"))
    run = tracelens("run", "--config", "bad.yaml", cwd=project, expect=2)
    assert "unknown key(s) under run: num_run" in run.stderr and run.stdout == ""

    # 12. Checkpoint and resume: the second run re-executes nothing.
    adapter.write_text(source.replace(
        TRUSTED,
        "from pathlib import Path\n"
        '    with Path("calls.log").open("a", encoding="utf-8") as log:\n'
        '        log.write(input_data["question"] + "\\n")\n'
        f"    {TRUSTED}",
    ))
    checkpoint_args = (
        "run", "--config", "tracelens.yaml", "--no-baseline-check", "--num-runs", "2",
        "--checkpoint", "eval/results/checkpoint.json",
        "--output", "eval/results/checkpoint-results.json",
    )
    tracelens(*checkpoint_args, cwd=project, expect=0)
    calls = (project / "calls.log").read_text().splitlines()
    assert len(calls) == 4  # 2 tasks x 2 runs
    assert load(project / "eval/results/checkpoint-results.json")["total_trials"] == 4
    tracelens(*checkpoint_args, cwd=project, expect=0)
    assert (project / "calls.log").read_text().splitlines() == calls  # nothing re-ran
    assert load(project / "eval/results/checkpoint-results.json")["total_trials"] == 4
    adapter.write_text(source)

    # 13. The whole suite passes again, and compare calls it equivalent.
    run = tracelens("run", "--config", "tracelens.yaml", cwd=project, expect=0)
    assert load(results)["gate"]["status"] == "passed"
    compare = tracelens(
        "compare", "eval/results/trusted-trials.json", "eval/results/trials.json",
        cwd=project, expect=0,
    )
    assert "equivalent within the practical threshold" in compare.stdout

    # 14. The saved artifacts are readable by the other documented commands.
    rendered = tracelens(
        "report", "--results", "eval/results/results.json", "--format", "markdown",
        cwd=project, expect=0,
    )
    assert "## Baseline Gate" in rendered.stdout and "## Run Provenance" in rendered.stdout
    tracelens(
        "sample", "--trials", "eval/results/trials.json", "--size", "2",
        "--output", "eval/results/review.json", cwd=project, expect=0,
    )
    worksheet = load(project / "eval/results/review.json")
    assert isinstance(worksheet, dict | list)
    tracelens("report", "--results", "eval/results/trials.json", cwd=project, expect=2)
