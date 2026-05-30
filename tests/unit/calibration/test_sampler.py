"""Tests for review-sample selection (the human-eval loop's `sample` step)."""

import pytest

from tracelens.calibration.sampler import (
    ReviewWorksheet,
    sample_for_review,
)
from tracelens.core.outcome import Outcome
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch, TrialStatus


def _trial(task_id: str, score: float, output: str = "out") -> Trial:
    """Build a completed, gradeable trial with a single outcome."""
    trial = Trial(task_id=task_id, status=TrialStatus.COMPLETED)
    trial.transcript = Transcript(task_id=task_id, final_output=output)
    trial.add_outcome(
        Outcome(
            trial_id=trial.trial_id,
            grader_id="g",
            passed=score >= 0.5,
            score=score,
        )
    )
    return trial


def _batch(scores: list[float]) -> TrialBatch:
    batch = TrialBatch()
    for i, s in enumerate(scores):
        batch.add_trial(_trial(f"task-{i}", s))
    return batch


def test_diverse_strategy_spans_the_score_range() -> None:
    # Scores clustered at the extremes plus a middle value.
    batch = _batch([0.0, 0.05, 0.1, 0.5, 0.9, 0.95, 1.0])

    sheet = sample_for_review(batch, size=3, strategy="diverse")

    assert isinstance(sheet, ReviewWorksheet)
    assert len(sheet.items) == 3
    selected_scores = sorted(item.grader_score for item in sheet.items)
    # A diverse sample must include the lowest and highest scoring trials,
    # not just whatever happens to be clustered together.
    assert selected_scores[0] == 0.0
    assert selected_scores[-1] == 1.0


def test_boundary_strategy_picks_scores_nearest_the_pass_threshold() -> None:
    batch = _batch([0.0, 0.48, 0.52, 1.0])

    sheet = sample_for_review(batch, size=2, strategy="boundary")

    selected = sorted(item.grader_score for item in sheet.items)
    # The two ambiguous trials straddling 0.5 are where grader/human
    # disagreement is most likely, so they should be surfaced first.
    assert selected == [0.48, 0.52]


def test_failures_strategy_returns_only_failing_trials() -> None:
    batch = _batch([0.2, 0.4, 0.8, 0.9])

    sheet = sample_for_review(batch, size=10, strategy="failures")

    assert sheet.items  # there are failures to find
    assert all(item.grader_passed is False for item in sheet.items)
    assert {item.grader_score for item in sheet.items} == {0.2, 0.4}


def test_size_larger_than_available_returns_all_without_error() -> None:
    batch = _batch([0.3, 0.7])

    sheet = sample_for_review(batch, size=50, strategy="diverse")

    assert len(sheet.items) == 2


def test_random_strategy_is_deterministic_for_a_given_seed() -> None:
    batch = _batch([i / 20 for i in range(20)])

    first = sample_for_review(batch, size=5, strategy="random", seed=42)
    second = sample_for_review(batch, size=5, strategy="random", seed=42)

    assert [i.trial_id for i in first.items] == [i.trial_id for i in second.items]


def test_ungradeable_trials_are_skipped() -> None:
    batch = _batch([0.6])
    # A trial with no transcript and no outcome cannot be reviewed.
    batch.add_trial(Trial(task_id="empty", status=TrialStatus.FAILED))

    sheet = sample_for_review(batch, size=10, strategy="diverse")

    assert len(sheet.items) == 1
    assert sheet.items[0].task_id == "task-0"


def test_worksheet_excerpt_is_truncated() -> None:
    batch = TrialBatch()
    batch.add_trial(_trial("long", 0.5, output="x" * 1000))

    sheet = sample_for_review(batch, size=1, strategy="diverse", excerpt_chars=50)

    assert len(sheet.items[0].output_excerpt) <= 50


def test_to_annotation_template_is_consumable_by_calibration() -> None:
    from tracelens.calibration.analyzer import AnnotationSet

    batch = _batch([0.3, 0.9])
    sheet = sample_for_review(batch, size=2, strategy="diverse")

    template = sheet.to_annotation_template()
    assert all("task_id" in row for row in template)
    assert all(row["human_score"] is None for row in template)

    # Once a human fills in the scores, the same shape loads as annotations.
    for row in template:
        row["human_score"] = 0.5
        row["human_passed"] = True
    annotations = AnnotationSet.from_json_list(template)
    assert len(annotations.annotations) == 2


def test_unknown_strategy_raises() -> None:
    batch = _batch([0.5])
    with pytest.raises(ValueError, match="strategy"):
        sample_for_review(batch, size=1, strategy="nonsense")
