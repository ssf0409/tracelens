"""Tests for batch- and report-level token usage aggregation.

Transcript already aggregates per-step tokens; these tests pin down the
roll-up across a TrialBatch and into ReportData, so eval cost is visible
without walking every transcript by hand.
"""

from tracelens.core.transcript import StepType, Transcript, TranscriptStep
from tracelens.core.trial import Trial, TrialBatch
from tracelens.reporting.generator import ReportGenerator


def _trial_with_tokens(task_id: str, tokens_in: int, tokens_out: int) -> Trial:
    transcript = Transcript(task_id=task_id)
    transcript.add_step(
        TranscriptStep(
            step_type=StepType.LLM_CALL,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )
    )
    return Trial(task_id=task_id, transcript=transcript)


def test_batch_token_totals() -> None:
    batch = TrialBatch()
    batch.add_trial(_trial_with_tokens("a", tokens_in=100, tokens_out=50))
    batch.add_trial(_trial_with_tokens("b", tokens_in=200, tokens_out=70))
    batch.add_trial(Trial(task_id="c"))  # no transcript → contributes 0

    assert batch.total_input_tokens == 300
    assert batch.total_output_tokens == 120
    assert batch.total_tokens == 420


def test_empty_batch_token_totals_are_zero() -> None:
    batch = TrialBatch()
    assert batch.total_input_tokens == 0
    assert batch.total_output_tokens == 0
    assert batch.total_tokens == 0


def test_report_includes_token_totals() -> None:
    batch = TrialBatch()
    batch.add_trial(_trial_with_tokens("a", tokens_in=10, tokens_out=5))

    report = ReportGenerator().build_report(batch)

    assert report.total_input_tokens == 10
    assert report.total_output_tokens == 5
    data = report.to_dict()
    assert data["total_input_tokens"] == 10
    assert data["total_output_tokens"] == 5
