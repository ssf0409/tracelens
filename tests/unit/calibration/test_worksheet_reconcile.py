"""Reconcile directly from a self-contained review worksheet.

The worksheet emitted by `sample_for_review` carries the grader outcome per row
alongside the (human-filled) grades, so calibration can pair them per-row
without a separate results file and without losing trials that share a task_id.
"""

from tracelens.calibration.analyzer import CalibrationAnalyzer
from tracelens.calibration.sampler import sample_for_review
from tracelens.core.outcome import Outcome
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch, TrialStatus


def _worksheet_row(task_id: str, grader: float, human: float | None) -> dict:
    return {
        "task_id": task_id,
        "grader_score": grader,
        "grader_passed": grader >= 0.5,
        "human_score": human,
        "human_passed": (human >= 0.5) if human is not None else None,
    }


def test_analyze_worksheet_pairs_grader_and_human_per_row() -> None:
    rows = [
        _worksheet_row("t0", 0.9, 0.5),
        _worksheet_row("t1", 0.8, 0.9),
        _worksheet_row("t2", 0.6, 0.2),
    ]

    result = CalibrationAnalyzer(threshold=0.7).analyze_worksheet(rows)

    assert result.sample_count == 3
    assert result.pearson_r is not None
    # t2: grader passed (0.6), human failed (0.2) -> a disagreement pair exists.
    assert any(not p.pass_agree for p in result.pairs)


def test_analyze_worksheet_skips_unfilled_rows() -> None:
    rows = [
        _worksheet_row("t0", 0.9, 0.8),
        _worksheet_row("t1", 0.7, None),  # reviewer left this blank
        _worksheet_row("t2", 0.4, 0.3),
    ]

    result = CalibrationAnalyzer().analyze_worksheet(rows)

    assert result.sample_count == 2


def test_analyze_worksheet_keeps_duplicate_task_ids_distinct() -> None:
    # Multi-run: two trials of the same task, graded differently by a human.
    rows = [
        _worksheet_row("same", 0.9, 0.9),
        _worksheet_row("same", 0.9, 0.2),
        _worksheet_row("other", 0.3, 0.3),
    ]

    result = CalibrationAnalyzer().analyze_worksheet(rows)

    # Both "same" rows are kept as distinct pairs (not collapsed by task_id).
    assert result.sample_count == 3


def test_sampler_worksheet_round_trips_into_analyze_worksheet() -> None:
    batch = TrialBatch()
    for i, s in enumerate([0.2, 0.55, 0.9]):
        trial = Trial(task_id=f"t{i}", status=TrialStatus.COMPLETED)
        trial.transcript = Transcript(task_id=f"t{i}", final_output="x")
        trial.add_outcome(Outcome(trial_id=trial.trial_id, grader_id="g", passed=s >= 0.5, score=s))
        batch.add_trial(trial)

    sheet = sample_for_review(batch, size=3, strategy="diverse")
    rows = sheet.to_annotation_template()
    # Worksheet must carry trial_id so a reviewer's grade ties back to the trial.
    assert all("trial_id" in row for row in rows)

    # Simulate a reviewer filling in the human columns, then reconcile.
    for row in rows:
        row["human_score"] = row["grader_score"]
        row["human_passed"] = row["grader_passed"]
    result = CalibrationAnalyzer().analyze_worksheet(rows)
    assert result.sample_count == 3
    assert result.pearson_r == 1.0  # human exactly matched grader
