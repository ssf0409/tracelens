"""Human-eval calibration on a recorded dataset — no API keys, no network.

LLM-as-judge graders drift. The only way to catch it is to compare the grader
against human judgement on a sample, periodically. This example shows the
reconciliation step on a *recorded* dataset so it runs anywhere:

1. A small recorded eval: each task has a transcript, an automated grader
   score, and a human grade.
2. ``sample_for_review`` picks which trials a human should grade (in a real
   workflow you would do this *before* collecting human grades).
3. ``CalibrationAnalyzer`` compares the grader against the human grades and
   reports correlation, agreement, and bias.
4. We surface the disagreement cases and recommend a concrete next step.

The grader here is deliberately *miscalibrated* (systematically generous), so
the example demonstrates the interesting path: drift detected, false-positive
disagreements, and a tuning recommendation.

Run it::

    python examples/human_eval_calibration.py
"""

from tracelens.calibration import (
    AnnotationSet,
    CalibrationAnalyzer,
    CalibrationResult,
    sample_for_review,
)
from tracelens.core.outcome import Outcome
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch, TrialStatus

# ---------------------------------------------------------------------------
# 1. The recorded dataset.
#
# Each row: (task_id, agent_output, grader_score, human_score).
# The grader passes everything >= 0.5 and scores generously; the human is
# stricter and ranks the answers differently. That mismatch is the whole point.
# ---------------------------------------------------------------------------
RECORDED = [
    # task    output                       grader  human
    ("t0", "Paris is the capital.", 0.90, 0.55),
    ("t1", "A thorough, sourced answer.", 0.80, 0.92),
    ("t2", "Confident but wrong.", 0.85, 0.20),  # false positive
    ("t3", "Mostly right, minor gap.", 0.70, 0.85),
    ("t4", "Plausible-sounding filler.", 0.60, 0.25),  # false positive
    ("t5", "Slick but unsupported.", 0.95, 0.45),  # false positive
    ("t6", "Correct and well-argued.", 0.75, 0.90),
    ("t7", "Verbose, evades the question.", 0.88, 0.40),  # false positive
]

PASS_THRESHOLD = 0.5


def build_batch() -> TrialBatch:
    """Reconstruct the recorded eval as a TrialBatch with grader outcomes."""
    batch = TrialBatch()
    for task_id, output, grader_score, _human in RECORDED:
        trial = Trial(task_id=task_id, status=TrialStatus.COMPLETED)
        trial.transcript = Transcript(task_id=task_id, final_output=output)
        trial.add_outcome(
            Outcome(
                trial_id=trial.trial_id,
                grader_id="quality-judge",
                passed=grader_score >= PASS_THRESHOLD,
                score=grader_score,
            )
        )
        batch.add_trial(trial)
    return batch


def human_annotations() -> AnnotationSet:
    """The human grades, as collected from the review worksheet."""
    return AnnotationSet.from_json_list(
        [
            {
                "task_id": task_id,
                "human_score": human,
                "human_passed": human >= PASS_THRESHOLD,
            }
            for task_id, _output, _grader, human in RECORDED
        ]
    )


def recommend_action(result: CalibrationResult) -> str:
    """Turn calibration metrics into a concrete next step.

    The advice depends on *how* the grader is wrong:
    - calibrated → nothing to do;
    - ranks well but biased → adjust the threshold / rescale scores;
    - ranking diverges from humans → the prompt/criteria need work.
    """
    if result.is_calibrated:
        return "Grader is calibrated (r >= threshold). No action needed."

    bias = result.grader_bias or 0.0
    corr = result.pearson_r if result.pearson_r is not None else 0.0

    if corr >= result.threshold * 0.8 and abs(bias) >= 0.1:
        direction = "generous" if bias > 0 else "harsh"
        return (
            f"Grader ranks answers reasonably (r={corr:.2f}) but is "
            f"systematically {direction} (bias={bias:+.2f}). Recalibrate the "
            f"pass threshold or rescale scores rather than rewriting the prompt."
        )
    return (
        f"Grader ranking diverges from humans (r={corr:.2f} < "
        f"{result.threshold}). Revise the grading prompt/criteria — start from "
        f"the disagreement cases below."
    )


def main() -> None:
    batch = build_batch()

    # 2. Which trials should a human grade? Pick the cases nearest the pass/fail
    #    line, where grader/human disagreement is likeliest. (Here we already
    #    have every grade, so this is illustrative of the selection step.)
    worksheet = sample_for_review(batch, size=4, strategy="boundary")

    # 3. Reconcile the automated grader against the human grades.
    grader_outcomes = {t.task_id: t.outcomes[0] for t in batch.trials}
    result = CalibrationAnalyzer(threshold=0.7).analyze(
        grader_outcomes, human_annotations()
    )

    print("\ntracelens human-eval calibration")
    print("--------------------------------")
    print(
        f"recorded trials : {len(batch.trials)}   "
        f"would review (boundary, n=4): "
        f"{', '.join(item.task_id for item in worksheet.items)}"
    )
    print()
    print(result.render_table())

    # 4. Surface disagreements: cases where grader and human reached a
    #    different pass/fail verdict. These are where to look first.
    print("\nDisagreements (grader vs human pass/fail)")
    print("-----------------------------------------")
    disagreements = [p for p in result.pairs if not p.pass_agree]
    for pair in disagreements:
        print(
            f"  {pair.task_id}: grader={'PASS' if pair.grader_passed else 'FAIL'} "
            f"({pair.grader_score:.2f})  "
            f"human={'PASS' if pair.human_passed else 'FAIL'} "
            f"({pair.human_score:.2f})  delta={pair.score_delta:+.2f}"
        )

    print(f"\nRecommended action:\n  {recommend_action(result)}")


if __name__ == "__main__":
    main()
