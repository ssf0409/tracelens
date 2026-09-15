"""Unit tests for the `tracelens calibrate` subcommand."""

import argparse
import json
from pathlib import Path
from typing import Any

from tracelens.cli.calibrate import cmd_calibrate
from tracelens.core.outcome import Outcome
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch, TrialStatus


def _write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data))
    return path


class DummyZeroArgGrader:
    """Grader class supporting zero-arg constructor."""

    def __init__(self) -> None:
        self.grader_id = "DummyZeroArgGrader"

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        return Outcome(
            trial_id="test-trial",
            grader_id=self.grader_id,
            passed=True,
            score=1.0,
        )


def test_calibrate_constant_scores_perfect_agreement(tmp_path: Path) -> None:
    # Constant grader and human scores: Pearson is undefined, but agreement is 1.0 -> exit 0
    worksheet = [
        {
            "task_id": f"t{i}",
            "grader_score": 1.0,
            "grader_passed": True,
            "human_score": 1.0,
            "human_passed": True,
        }
        for i in range(5)
    ]
    wf = _write_json(tmp_path / "review.json", worksheet)

    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
        import_root=None,
    )
    rc = cmd_calibrate(args)
    assert rc == 0


def test_calibrate_constant_scores_disagreement(tmp_path: Path) -> None:
    # Constant grader scores (1.0), human scores fail -> pass_fail_agreement is 0.0 -> exit 1
    worksheet = [
        {
            "task_id": f"t{i}",
            "grader_score": 1.0,
            "grader_passed": True,
            "human_score": 0.0,
            "human_passed": False,
        }
        for i in range(5)
    ]
    wf = _write_json(tmp_path / "review.json", worksheet)

    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
        import_root=None,
    )
    rc = cmd_calibrate(args)
    assert rc == 1


def test_calibrate_malformed_annotations_file(tmp_path: Path) -> None:
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
        import_root=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_missing_annotations_file() -> None:
    args = argparse.Namespace(
        annotations="nonexistent.json",
        results=None,
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
        import_root=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_empty_worksheet(tmp_path: Path) -> None:
    wf = _write_json(tmp_path / "empty.json", [])
    args = argparse.Namespace(
        annotations=str(wf),
        results=None,
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=None,
        import_root=None,
    )
    assert cmd_calibrate(args) == 2


def test_calibrate_with_run_results_document(tmp_path: Path) -> None:
    # Test --results pointing to a results.json produced by tracelens run (with task_summaries)
    annotations = [
        {"task_id": "t1", "human_score": 1.0, "human_passed": True},
        {"task_id": "t2", "human_score": 0.0, "human_passed": False},
    ]
    ann_file = _write_json(tmp_path / "annotations.json", annotations)

    run_results = {
        "total_trials": 2,
        "total_tasks": 2,
        "task_summaries": [
            {"task_id": "t1", "mean_score": 1.0, "pass_rate": 1.0},
            {"task_id": "t2", "mean_score": 0.0, "pass_rate": 0.0},
        ],
    }
    results_file = _write_json(tmp_path / "results.json", run_results)

    out_file = tmp_path / "calib_out.json"
    args = argparse.Namespace(
        annotations=str(ann_file),
        results=str(results_file),
        transcripts=None,
        grader=None,
        samples=None,
        threshold=0.7,
        output=str(out_file),
        import_root=None,
    )
    rc = cmd_calibrate(args)
    assert rc == 0
    assert out_file.exists()


def test_calibrate_with_trials_json_and_zero_arg_grader(tmp_path: Path) -> None:
    # Test --transcripts with a TrialBatch (trials.json) and a custom grader
    batch = TrialBatch()
    trial = Trial(task_id="t1", status=TrialStatus.COMPLETED)
    trial.transcript = Transcript(task_id="t1", final_output="output 1")
    batch.add_trial(trial)
    trials_file = _write_json(tmp_path / "trials.json", batch.to_dict())

    samples = [
        {
            "task_id": "t1",
            "name": "Task 1",
            "input_data": {"q": "hello"},
        }
    ]
    samples_file = _write_json(tmp_path / "samples.json", samples)

    annotations = [
        {"task_id": "t1", "human_score": 1.0, "human_passed": True},
        {"task_id": "t1", "human_score": 1.0, "human_passed": True},
    ]
    ann_file = _write_json(tmp_path / "annotations.json", annotations)

    args = argparse.Namespace(
        annotations=str(ann_file),
        results=None,
        transcripts=str(trials_file),
        grader="tests.unit.cli.test_calibrate.DummyZeroArgGrader",
        samples=str(samples_file),
        threshold=0.7,
        output=None,
        import_root=str(Path.cwd()),
    )
    rc = cmd_calibrate(args)
    assert rc == 0
