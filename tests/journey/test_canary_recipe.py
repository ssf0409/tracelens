"""An actual downstream Python gate, run as separate installed-package processes."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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


def moved(gate: dict[str, Any]) -> set[str]:
    """Metrics that differ from the reference: absent on one side, or a nonzero delta."""
    return {
        name for name, row in gate["comparisons"].items()
        if row["coverage"] != "matched" or row["delta"] != 0
    }


# Each fault moves only the metrics it is named for, so a blocked gate is
# attributable to that signal rather than to a side effect of the injection.
@pytest.mark.parametrize("scenario,exit_code,missing,moved_metrics", [
    ("unchanged", 0, {}, set()),
    ("missing", 1, {"accepted": "missing_from_current"}, {"accepted"}),
    ("new", 1, {"rejected.permission": "missing_from_baseline"}, {"rejected.permission"}),
    ("empty", 1, {"accepted": "missing_from_current"},
     {"accepted", "rejected.invalid", "rejected.quota"}),
])
def test_replay_coverage_gate(
    recorded_canary: Path, scenario: str, exit_code: int, missing: dict[str, str],
    moved_metrics: set[str],
) -> None:
    project = recorded_canary
    frozen = (project / "baseline.json").read_bytes()
    invoke(project, "check", "--baseline", "baseline.json", "--scenario", scenario,
           "--output-dir", scenario, expect=exit_code)
    gate = json.loads((project / scenario / "gate.json").read_text())
    assert moved(gate) == moved_metrics
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


@pytest.mark.parametrize("scenario,count", [("increase", 3), ("decrease", 1)])
def test_value_changes_in_either_direction_block(recorded_canary: Path, scenario: str, count: int) -> None:
    project = recorded_canary
    frozen = (project / "baseline.json").read_bytes()
    invoke(project, "check", "--baseline", "baseline.json", "--scenario", scenario,
           "--output-dir", scenario, expect=1)
    gate = json.loads((project / scenario / "gate.json").read_text())
    row = gate["comparisons"]["accepted"]
    assert row["coverage"] == "matched"
    assert row["current"] == count
    assert row["changed"] is True
    assert row["regression"] is False
    assert gate["status"] == "blocked"
    assert (project / "baseline.json").read_bytes() == frozen


@pytest.mark.parametrize("invalid", [[], {"fixed-workflow": []}, {"fixed-workflow": {"metrics": []}}])
def test_invalid_baseline_structure_is_an_input_error(recorded_canary: Path, invalid: object) -> None:
    project = recorded_canary
    (project / "baseline.json").write_text(json.dumps(invalid))
    invoke(project, "check", "--baseline", "baseline.json", "--output-dir", "bad-shape", expect=2)
    assert not (project / "bad-shape" / "trials.json").exists()


def test_a_known_zero_counter_change_persists_a_blocked_gate(recorded_canary: Path) -> None:
    project = recorded_canary
    frozen = (project / "baseline.json").read_bytes()
    invoke(project, "check", "--baseline", "baseline.json", "--scenario", "zero-channel",
           "--output-dir", "zero-channel", expect=1)
    gate = json.loads((project / "zero-channel" / "gate.json").read_text())
    row = gate["comparisons"]["rejected.quota"]
    assert row["baseline"] == 0
    assert row["current"] == 1
    assert row["changed"] is True
    assert row["relative_change"] is None
    assert row["unavailable_fields"] == ["relative_change"]
    assert gate["status"] == "blocked"
    assert gate["exit_code"] == 1
    assert (project / "zero-channel" / "trials.json").is_file()
    assert (project / "baseline.json").read_bytes() == frozen
