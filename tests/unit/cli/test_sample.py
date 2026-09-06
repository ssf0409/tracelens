"""Tests for the `tracelens sample` command and the `reconcile` alias."""

import argparse
import json
from pathlib import Path

import pytest

from tracelens.cli.calibrate import cmd_calibrate
from tracelens.cli.main import build_parser
from tracelens.cli.sample import cmd_sample
from tracelens.core.outcome import Outcome
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch, TrialStatus


def _trials_file(tmp_path: Path, scores: list[float]) -> Path:
    batch = TrialBatch()
    for i, s in enumerate(scores):
        trial = Trial(task_id=f"task-{i}", status=TrialStatus.COMPLETED)
        trial.transcript = Transcript(task_id=f"task-{i}", final_output=f"answer-{i}")
        trial.add_outcome(
            Outcome(trial_id=trial.trial_id, grader_id="g", passed=s >= 0.5, score=s)
        )
        batch.add_trial(trial)
    path = tmp_path / "trials.json"
    path.write_text(json.dumps(batch.to_dict()))
    return path


def test_sample_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["sample", "--trials", "trials.json", "--size", "20"])
    assert args.command == "sample"
    assert args.trials == "trials.json"
    assert args.size == 20
    assert args.strategy == "diverse"
    assert args.seed == 0


def test_reconcile_is_an_alias_for_calibrate() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "reconcile",
        "--grader", "my.Grader",
        "--samples", "samples.json",
        "--annotations", "human.json",
        "--results", "results.json",
    ])
    assert args.command == "reconcile"
    # Same options as calibrate, so the existing handler can serve it.
    assert args.grader == "my.Grader"
    assert args.annotations == "human.json"


def test_cmd_sample_writes_a_fillable_worksheet(tmp_path: Path) -> None:
    trials = _trials_file(tmp_path, [0.1, 0.5, 0.9])
    out = tmp_path / "review.json"

    args = argparse.Namespace(
        trials=str(trials),
        size=2,
        strategy="diverse",
        seed=0,
        excerpt_chars=280,
        output=str(out),
    )
    rc = cmd_sample(args)

    assert rc == 0
    rows = json.loads(out.read_text())
    assert len(rows) == 2
    assert all(row["human_score"] is None for row in rows)
    assert all("task_id" in row for row in rows)


def test_cmd_sample_warns_when_no_gradeable_trials(
    tmp_path: Path, capsys: "pytest.CaptureFixture[str]"
) -> None:
    # A valid JSON that yields no gradeable trials (e.g. the wrong file, or an
    # empty batch) should warn rather than silently emit an empty worksheet.
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(TrialBatch().to_dict()))

    args = argparse.Namespace(
        trials=str(empty),
        size=5,
        strategy="diverse",
        seed=0,
        excerpt_chars=280,
        output=None,
    )
    rc = cmd_sample(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "no gradeable trials" in captured.err


def test_cmd_sample_handles_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")

    args = argparse.Namespace(
        trials=str(bad), size=2, strategy="diverse", seed=0,
        excerpt_chars=280, output=None,
    )
    # A usage/input error: exit 2, not an unhandled exception.
    assert cmd_sample(args) == 2


def test_cmd_reconcile_from_self_contained_worksheet(tmp_path: Path) -> None:
    # A filled worksheet carries grader + human grades per row, so reconcile
    # needs no --results/--transcripts/--grader/--samples.
    rows = [
        {
            "task_id": f"t{i}",
            "trial_id": f"x{i}",
            "grader_score": g,
            "grader_passed": g >= 0.5,
            "human_score": h,
            "human_passed": h >= 0.5,
        }
        for i, (g, h) in enumerate(
            [(0.9, 0.85), (0.8, 0.9), (0.6, 0.55), (0.7, 0.75), (0.4, 0.45)]
        )
    ]
    wf = tmp_path / "review.json"
    wf.write_text(json.dumps(rows))

    args = argparse.Namespace(
        grader=None,
        samples=None,
        annotations=str(wf),
        transcripts=None,
        results=None,
        threshold=0.7,
        output=None,
    )
    rc = cmd_calibrate(args)

    # Grader tracks human closely here -> calibrated -> exit 0.
    assert rc == 0
