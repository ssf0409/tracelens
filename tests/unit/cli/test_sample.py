"""Tests for the `tracelens sample` command and the `reconcile` alias."""

import argparse
import json
from pathlib import Path

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
