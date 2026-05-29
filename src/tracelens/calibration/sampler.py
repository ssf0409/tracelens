"""Select a subset of trials for human review (the `sample` step).

Human calibration only works if a person actually grades some transcripts, and
nobody hand-grades a thousand of them. This module picks a small, useful subset
and emits a worksheet that, once a human fills in the scores, loads directly as
an :class:`~tracelens.calibration.analyzer.AnnotationSet` for ``reconcile``.

The sampling *strategy* is the whole point — a random handful tells you little,
but a sample that spans the score range (``diverse``) or sits on the pass/fail
boundary (``boundary``) is where grader/human disagreement actually shows up.
"""

import random

from pydantic import BaseModel, Field

from tracelens.core.trial import Trial, TrialBatch

# Normalized pass/fail threshold. Outcome scores are 0-1 (see Outcome.score),
# so 0.5 is the canonical decision boundary used for the "boundary" strategy.
_PASS_THRESHOLD = 0.5

STRATEGIES = ("diverse", "boundary", "failures", "random")


class ReviewItem(BaseModel):
    """One trial presented for human review.

    The ``human_*`` fields are left blank for a reviewer to fill in. The
    remaining fields are read-only context to make grading possible.
    """

    task_id: str
    trial_id: str
    grader_score: float
    grader_passed: bool
    output_excerpt: str

    # Filled in by the human reviewer.
    human_score: float | None = None
    human_passed: bool | None = None
    notes: str = ""


class ReviewWorksheet(BaseModel):
    """A set of trials selected for human review."""

    strategy: str
    requested_size: int
    source_batch_id: str | None = None
    items: list[ReviewItem] = Field(default_factory=list)

    def to_annotation_template(self) -> list[dict[str, object]]:
        """Return rows shaped like the annotations file ``reconcile`` expects.

        Each row carries the blank ``human_score``/``human_passed`` for a
        reviewer to complete, plus read-only ``grader_*`` context that the
        annotation loader ignores.
        """
        return [
            {
                "task_id": item.task_id,
                "human_score": item.human_score,
                "human_passed": item.human_passed,
                "notes": item.notes,
                "grader_score": item.grader_score,
                "grader_passed": item.grader_passed,
                "output_excerpt": item.output_excerpt,
            }
            for item in self.items
        ]


def _gradeable_trials(batch: TrialBatch) -> list[Trial]:
    """Trials that have both a transcript and a grader score to review."""
    return [
        t
        for t in batch.trials
        if t.transcript is not None and t.aggregate_score is not None
    ]


def _excerpt(trial: Trial, max_chars: int) -> str:
    if trial.transcript is None or trial.transcript.final_output is None:
        return ""
    return str(trial.transcript.final_output)[:max_chars]


def _diverse(trials: list[Trial], size: int) -> list[Trial]:
    """Pick `size` trials spread evenly across the sorted score range."""
    ordered = sorted(trials, key=lambda t: t.aggregate_score or 0.0)
    if size >= len(ordered):
        return ordered
    # Evenly spaced indices including both endpoints (min and max score).
    step = (len(ordered) - 1) / (size - 1) if size > 1 else 0.0
    indices = sorted({round(i * step) for i in range(size)})
    return [ordered[i] for i in indices]


def _boundary(trials: list[Trial], size: int) -> list[Trial]:
    """Pick the `size` trials whose score is closest to the pass threshold."""
    ordered = sorted(
        trials, key=lambda t: abs((t.aggregate_score or 0.0) - _PASS_THRESHOLD)
    )
    return ordered[:size]


def _failures(trials: list[Trial], size: int) -> list[Trial]:
    """Pick failing trials, lowest score first."""
    failing = [t for t in trials if not t.passed]
    failing.sort(key=lambda t: t.aggregate_score or 0.0)
    return failing[:size]


def _random(trials: list[Trial], size: int, seed: int) -> list[Trial]:
    """Pick a reproducible random sample.

    Sort by trial_id first so the selection depends only on the seed, not on
    the (arbitrary) order trials happen to sit in the batch.
    """
    ordered = sorted(trials, key=lambda t: t.trial_id)
    rng = random.Random(seed)
    if size >= len(ordered):
        return ordered
    return rng.sample(ordered, size)


def sample_for_review(
    batch: TrialBatch,
    size: int,
    strategy: str = "diverse",
    seed: int = 0,
    excerpt_chars: int = 280,
) -> ReviewWorksheet:
    """Select trials from `batch` for human review.

    Args:
        batch: The trials to sample from (e.g. loaded from ``--save-trials``).
        size: Maximum number of trials to select. Fewer are returned when the
            batch has fewer gradeable trials than requested.
        strategy: One of ``diverse`` (span the score range — the default and
            best general choice), ``boundary`` (cases nearest the pass/fail
            line, where disagreement is likeliest), ``failures`` (failing
            trials only), or ``random`` (reproducible random sample).
        seed: Seed for the ``random`` strategy, for reproducible worksheets.
        excerpt_chars: Max characters of each trial's final output to include.

    Returns:
        A :class:`ReviewWorksheet` whose items have blank human-score fields.

    Raises:
        ValueError: If `strategy` is not a known strategy.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown sampling strategy {strategy!r}; choose from {STRATEGIES}"
        )

    trials = _gradeable_trials(batch)

    if strategy == "diverse":
        selected = _diverse(trials, size)
    elif strategy == "boundary":
        selected = _boundary(trials, size)
    elif strategy == "failures":
        selected = _failures(trials, size)
    else:  # "random"
        selected = _random(trials, size, seed)

    items = [
        ReviewItem(
            task_id=t.task_id,
            trial_id=t.trial_id,
            grader_score=t.aggregate_score or 0.0,
            grader_passed=t.passed,
            output_excerpt=_excerpt(t, excerpt_chars),
        )
        for t in selected
    ]

    return ReviewWorksheet(
        strategy=strategy,
        requested_size=size,
        source_batch_id=batch.batch_id,
        items=items,
    )
