"""An actual downstream Python gate, run as separate installed-package processes."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
PYTHON = os.environ.get("TRACELENS_CANARY_PYTHON", sys.executable)


def invoke(project: Path, *args: str, expect: int) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [PYTHON, str(project / "deterministic_canary.py"), *args], cwd=project,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == expect, result.stdout + result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    return result


@pytest.fixture
def recorded_canary(tmp_path: Path) -> Path:
    # Copy the recipe to a downstream project: no imports from repository source.
    for name in ("deterministic_canary.py", "canary_replay.json"):
        shutil.copyfile(EXAMPLES / name, tmp_path / name)
    invoke(tmp_path, "record", "--baseline", "baseline.json", "--output-dir", "record", expect=0)
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    assert baseline["fixed-workflow"]["baseline_type"] == "canary"
    assert baseline["fixed-workflow"]["promotion_policy"]["allow_auto_promotion"] is False
    assert baseline["fixed-workflow"]["metrics"]["accepted"]["baseline_value"] == 2
    return tmp_path


@pytest.mark.parametrize("scenario,exit_code,missing", [
    ("unchanged", 0, {}),
    ("missing", 1, {"accepted": "missing_from_current"}),
    ("new", 1, {"rejected.permission": "missing_from_baseline"}),
    ("empty", 1, {"accepted": "missing_from_current"}),
])
def test_replay_coverage_gate(
    recorded_canary: Path, scenario: str, exit_code: int, missing: dict[str, str],
) -> None:
    project = recorded_canary
    frozen = (project / "baseline.json").read_bytes()
    invoke(project, "check", "--baseline", "baseline.json", "--scenario", scenario,
           "--output-dir", scenario, expect=exit_code)
    gate = json.loads((project / scenario / "gate.json").read_text())
    assert gate["exit_code"] == exit_code
    assert gate["status"] == ("passed" if exit_code == 0 else "blocked")
    assert gate["coverage"] == "strict"
    for name, marker in missing.items():
        assert gate["comparisons"][name]["coverage"] == marker
        assert gate["comparisons"][name]["regression"] is None
    trials = json.loads((project / scenario / "trials.json").read_text())
    trial = trials["trials"][0]
    assert trial["transcript"]["steps"]
    assert gate["metrics"] == trial["outcomes"][0]["metrics"]
    assert gate["task_hash"] == json.loads((project / "record" / "gate.json").read_text())["task_hash"]
    assert (project / "baseline.json").read_bytes() == frozen


def test_corrupt_baseline_fails_before_execution(recorded_canary: Path) -> None:
    project = recorded_canary
    (project / "baseline.json").write_text("broken JSON")
    invoke(project, "check", "--baseline", "baseline.json", "--output-dir", "invalid", expect=2)
    assert not (project / "invalid" / "trials.json").exists()


def test_record_cannot_replace_the_frozen_reference(recorded_canary: Path) -> None:
    project = recorded_canary
    frozen = (project / "baseline.json").read_bytes()
    invoke(project, "record", "--baseline", "baseline.json", "--output-dir", "record2", expect=2)
    assert (project / "baseline.json").read_bytes() == frozen


def test_a_changed_fixture_is_not_the_same_replay(recorded_canary: Path) -> None:
    project = recorded_canary
    path = project / "canary_replay.json"
    data = json.loads(path.read_text())
    data["tasks"][0]["input_data"]["requests"].append({"id": "d", "approved": True})
    path.write_text(json.dumps(data))
    invoke(project, "check", "--baseline", "baseline.json", "--output-dir", "changed-task", expect=2)
    assert not (project / "changed-task" / "trials.json").exists()
