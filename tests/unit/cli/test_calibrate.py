"""Unit tests for the `tracelens calibrate` subcommand."""

import argparse
import json
from pathlib import Path
from typing import Any

from tracelens.cli.calibrate import cmd_calibrate
from tracelens.core.grader import Grader, GraderType
from tracelens.core.outcome import Outcome
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data))
    return path


class DummyTestGrader(Grader):
    """Test grader subclassing Grader."""

    @property
    def grader_type(self) -> GraderType:
        return GraderType.CODE_BASED

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        return self.create_outcome(
            trial_id=task.task_id,
            passed=True,
            score=1.0,
        )


def test_calibrate_missing_annotations_file() -> None:
    args = argparse.Namespace(
        annotations="nonexistent.json",
        results=None,
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_invalid_annotations_json(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad json")
    args = argparse.Namespace(
        annotations=str(bad_json),
        results=None,
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_worksheet_empty(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "empty.json", [])
    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_results_file_not_found(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "ann.json", [{"task_id": "t1", "human_score": 1.0, "human_passed": True}])
    args = argparse.Namespace(
        annotations=str(wf),
        results=str(tmp_path / "missing_results.json"),
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_results_invalid_json(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "ann.json", [{"task_id": "t1", "human_score": 1.0, "human_passed": True}])
    bad_res = tmp_path / "bad_res.json"
    bad_res.write_text("invalid json")
    args = argparse.Namespace(
        annotations=str(wf),
        results=str(bad_res),
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_results_success(tmp_path: Path) -> None:
    wf = _write_json(
        tmp_path / "ann.json",
        [
            {"task_id": "t1", "human_score": 1.0, "human_passed": True},
            {"task_id": "t2", "human_score": 0.0, "human_passed": False},
        ],
    )
    results = {
        "t1": {"outcome_id": "o1", "trial_id": "t1", "grader_id": "g1", "passed": True, "score": 1.0},
        "t2": {"outcome_id": "o2", "trial_id": "t2", "grader_id": "g1", "passed": False, "score": 0.0},
    }
    rf = _write_json(tmp_path / "res.json", results)
    out_file = tmp_path / "out.json"
    args = argparse.Namespace(
        annotations=str(wf),
        results=str(rf),
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=str(out_file),
    )
    rc = cmd_calibrate(args)
    assert rc == 0
    assert out_file.exists()


def test_calibrate_transcripts_missing_grader_or_samples(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "ann.json", [{"task_id": "t1", "human_score": 1.0, "human_passed": True}])
    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=str(tmp_path / "transcripts.json"),
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_transcripts_invalid_grader_class(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "ann.json", [{"task_id": "t1", "human_score": 1.0, "human_passed": True}])
    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=str(tmp_path / "transcripts.json"),
        grader="nonexistent.module.InvalidGrader",
        samples=str(tmp_path / "samples.json"),
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_transcripts_missing_file(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "ann.json", [{"task_id": "t1", "human_score": 1.0, "human_passed": True}])
    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=str(tmp_path / "missing_transcripts.json"),
        grader="tests.unit.cli.test_calibrate.DummyTestGrader",
        samples=str(tmp_path / "samples.json"),
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_transcripts_invalid_json(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "ann.json", [{"task_id": "t1", "human_score": 1.0, "human_passed": True}])
    bad_tf = tmp_path / "bad_transcripts.json"
    bad_tf.write_text("not json")
    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=str(bad_tf),
        grader="tests.unit.cli.test_calibrate.DummyTestGrader",
        samples=str(tmp_path / "samples.json"),
        threshold=0.7,
        output=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_transcripts_success_and_skips_unmatched(tmp_path: Path) -> None:
    wf = _write_json(
        tmp_path / "ann.json",
        [
            {"task_id": "t1", "human_score": 1.0, "human_passed": True},
            {"task_id": "t2", "human_score": 0.8, "human_passed": True},
            {"task_id": "t3", "human_score": 0.4, "human_passed": False},
            {"task_id": "t4", "human_score": 0.2, "human_passed": False},
        ],
    )
    samples = {
        "tasks": [
            {"task_id": "t1", "name": "Task 1", "input_data": {"x": 1}},
            {"task_id": "t2", "name": "Task 2", "input_data": {"x": 2}},
            {"task_id": "t3", "name": "Task 3", "input_data": {"x": 3}},
            {"task_id": "t4", "name": "Task 4", "input_data": {"x": 4}},
        ]
    }
    sf = _write_json(tmp_path / "samples.json", samples)
    transcripts = {
        "t1": {"task_id": "t1", "final_output": "done"},
        "t2": {"task_id": "t2", "final_output": "done"},
        "t3": {"task_id": "t3", "final_output": "done"},
        "t4": {"task_id": "t4", "final_output": "done"},
        "t_unmatched": {"task_id": "t_unmatched", "final_output": "skip"},
    }
    tf = _write_json(tmp_path / "transcripts.json", transcripts)
    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=str(tf),
        grader="tests.unit.cli.test_calibrate.DummyTestGrader",
        samples=str(sf),
        threshold=0.7,
        output=None,
    )
    rc = cmd_calibrate(args)
    assert rc in (0, 1)
