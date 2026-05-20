"""Outcome models for grading results.

An Outcome represents the result of grading a Trial - whether it passed,
the score achieved, and detailed metrics.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tracelens.core._time import utc_now


class GradeLevel(str, Enum):
    """Categorical grade levels for human-readable results."""

    EXCELLENT = "excellent"  # Score >= 0.9
    GOOD = "good"            # Score >= 0.7
    ACCEPTABLE = "acceptable"  # Score >= 0.5
    POOR = "poor"            # Score >= 0.3
    FAIL = "fail"            # Score < 0.3

    @classmethod
    def from_score(cls, score: float) -> "GradeLevel":
        """Convert a normalized score to a grade level."""
        if score >= 0.9:
            return cls.EXCELLENT
        elif score >= 0.7:
            return cls.GOOD
        elif score >= 0.5:
            return cls.ACCEPTABLE
        elif score >= 0.3:
            return cls.POOR
        return cls.FAIL


class Outcome(BaseModel):
    """Result of grading a trial.

    An Outcome contains:
    - Primary pass/fail determination
    - Normalized score (0-1)
    - Detailed metrics dict (grader-specific)
    - Optional feedback and reasoning

    Example:
        outcome = Outcome(
            trial_id="...",
            grader_id="quality",
            passed=True,
            score=0.85,
            metrics={"specificity": 0.9, "personalization": 0.8},
            feedback="Tasks are specific but could use more context references",
        )
    """

    outcome_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trial_id: str
    grader_id: str

    # Primary result
    passed: bool
    score: float = Field(ge=0.0, le=1.0)  # Normalized 0-1

    # Detailed metrics (grader-specific)
    metrics: dict[str, float] = Field(default_factory=dict)

    # Categorical grade (computed from score)
    grade_level: GradeLevel | None = None

    # Explanation/feedback
    feedback: str | None = None
    reasoning: str | None = None  # For LLM graders - the raw reasoning

    # Confidence (for non-deterministic graders)
    confidence: float | None = None

    # Timing
    graded_at: datetime = Field(default_factory=utc_now)
    grading_duration_ms: float | None = None

    # For human grades
    human_grader_id: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Set grade_level from score if not provided."""
        if self.grade_level is None:
            self.grade_level = GradeLevel.from_score(self.score)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a summary suitable for reporting."""
        return {
            "passed": self.passed,
            "score": self.score,
            "grade_level": self.grade_level.value if self.grade_level else None,
            "metrics": self.metrics,
            "feedback": self.feedback,
        }

    def to_ci_dict(self) -> dict[str, Any]:
        """Return a compact dict for CI output."""
        return {
            "grader": self.grader_id,
            "passed": self.passed,
            "score": round(self.score, 4),
            "grade": self.grade_level.value if self.grade_level else None,
        }


class AggregatedOutcome(BaseModel):
    """Aggregated outcomes across multiple trials or graders.

    Used for computing suite-level statistics like pass rate,
    mean score, and pass@k.
    """

    # Trial-level aggregation
    total_trials: int = 0
    passed_trials: int = 0
    failed_trials: int = 0

    # Score statistics
    mean_score: float = 0.0
    std_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    median_score: float = 0.0

    # Pass rates
    pass_rate: float = 0.0

    # Per-metric statistics
    metric_means: dict[str, float] = Field(default_factory=dict)
    metric_stds: dict[str, float] = Field(default_factory=dict)

    # Per-grader pass rates
    grader_pass_rates: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_outcomes(cls, outcomes: list[Outcome]) -> "AggregatedOutcome":
        """Compute aggregated statistics from a list of outcomes."""
        if not outcomes:
            return cls()

        scores = [o.score for o in outcomes]
        passed_count = sum(1 for o in outcomes if o.passed)

        # Collect metrics
        all_metrics: dict[str, list[float]] = {}
        grader_results: dict[str, list[bool]] = {}

        for o in outcomes:
            for metric, value in o.metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)

            if o.grader_id not in grader_results:
                grader_results[o.grader_id] = []
            grader_results[o.grader_id].append(o.passed)

        return cls(
            total_trials=len(outcomes),
            passed_trials=passed_count,
            failed_trials=len(outcomes) - passed_count,
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            median_score=float(np.median(scores)),
            pass_rate=passed_count / len(outcomes),
            metric_means={k: float(np.mean(v)) for k, v in all_metrics.items()},
            metric_stds={k: float(np.std(v)) for k, v in all_metrics.items()},
            grader_pass_rates={
                k: sum(v) / len(v) for k, v in grader_results.items()
            },
        )
