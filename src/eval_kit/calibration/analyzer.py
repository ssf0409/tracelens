"""Calibration analysis — compare grader scores against human annotations.

Computes correlation metrics (Pearson, Spearman), agreement (Cohen's kappa),
and bias detection to determine if an LLM grader is drifting from human judgment.
"""

from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats


class HumanAnnotation(BaseModel):
    """A single human annotation for a graded sample."""

    task_id: str
    human_score: float  # 0.0 - 1.0
    human_passed: bool
    notes: str | None = None


class AnnotationSet(BaseModel):
    """Collection of human annotations."""

    annotations: list[HumanAnnotation] = Field(default_factory=list)

    @classmethod
    def from_json_list(cls, data: list[dict[str, Any]]) -> "AnnotationSet":
        """Create from a list of annotation dicts."""
        return cls(annotations=[HumanAnnotation(**item) for item in data])

    def get_by_task_id(self, task_id: str) -> HumanAnnotation | None:
        for a in self.annotations:
            if a.task_id == task_id:
                return a
        return None


class CalibrationPair(BaseModel):
    """Paired grader/human comparison for a single sample."""

    task_id: str
    grader_score: float
    grader_passed: bool
    human_score: float
    human_passed: bool
    score_delta: float  # grader - human
    pass_agree: bool


class CalibrationResult(BaseModel):
    """Result of calibration analysis between grader and human annotations."""

    pairs: list[CalibrationPair] = Field(default_factory=list)
    sample_count: int = 0

    # Correlation metrics
    pearson_r: float | None = None
    spearman_rho: float | None = None

    # Agreement metrics
    pass_fail_agreement: float | None = None
    cohens_kappa: float | None = None

    # Score comparison
    mean_score_delta: float | None = None
    mae: float | None = None
    grader_bias: float | None = None  # positive = grader scores higher

    # Verdict
    threshold: float = 0.7
    is_calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "pearson_r": self.pearson_r,
            "spearman_rho": self.spearman_rho,
            "pass_fail_agreement": self.pass_fail_agreement,
            "cohens_kappa": self.cohens_kappa,
            "mean_score_delta": self.mean_score_delta,
            "mae": self.mae,
            "grader_bias": self.grader_bias,
            "threshold": self.threshold,
            "is_calibrated": self.is_calibrated,
            "pairs": [p.model_dump() for p in self.pairs],
        }

    def render_table(self) -> str:
        """Render a text summary table."""
        lines = [
            "Calibration Report",
            "=" * 50,
            f"Samples:              {self.sample_count}",
            f"Pearson r:            {self._fmt(self.pearson_r)}",
            f"Spearman rho:         {self._fmt(self.spearman_rho)}",
            f"Pass/Fail agreement:  {self._fmt(self.pass_fail_agreement)}",
            f"Cohen's kappa:        {self._fmt(self.cohens_kappa)}",
            f"Mean score delta:     {self._fmt(self.mean_score_delta)}",
            f"MAE:                  {self._fmt(self.mae)}",
            f"Grader bias:          {self._fmt(self.grader_bias)}",
            "-" * 50,
            f"Threshold:            {self.threshold}",
            f"Calibrated:           {'YES' if self.is_calibrated else 'NO - DRIFT DETECTED'}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fmt(val: float | None) -> str:
        return f"{val:.4f}" if val is not None else "N/A"


class CalibrationAnalyzer:
    """Compares grader outcomes against human annotations.

    Example:
        analyzer = CalibrationAnalyzer(threshold=0.7)
        result = analyzer.analyze(grader_outcomes, annotations)
        if not result.is_calibrated:
            print("Grader drift detected!")
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def analyze(
        self,
        grader_outcomes: dict[str, "OutcomeLike"],
        annotations: AnnotationSet,
    ) -> CalibrationResult:
        """Compare grader outcomes against human annotations.

        Args:
            grader_outcomes: Dict mapping task_id to an object with
                             `.score` (float) and `.passed` (bool) attributes.
            annotations: Human annotation set.

        Returns:
            CalibrationResult with correlation and agreement metrics.
        """
        pairs: list[CalibrationPair] = []

        for annotation in annotations.annotations:
            outcome = grader_outcomes.get(annotation.task_id)
            if outcome is None:
                continue

            grader_score = outcome.score
            grader_passed = outcome.passed

            pairs.append(CalibrationPair(
                task_id=annotation.task_id,
                grader_score=grader_score,
                grader_passed=grader_passed,
                human_score=annotation.human_score,
                human_passed=annotation.human_passed,
                score_delta=grader_score - annotation.human_score,
                pass_agree=grader_passed == annotation.human_passed,
            ))

        if len(pairs) < 2:
            return CalibrationResult(
                pairs=pairs,
                sample_count=len(pairs),
                threshold=self.threshold,
                is_calibrated=False,
            )

        grader_scores = np.array([p.grader_score for p in pairs])
        human_scores = np.array([p.human_score for p in pairs])
        deltas = grader_scores - human_scores

        # Pearson correlation
        pearson_r: float | None = None
        if np.std(grader_scores) > 0 and np.std(human_scores) > 0:
            pearson_r = float(np.corrcoef(grader_scores, human_scores)[0, 1])

        # Spearman rank correlation
        spearman_rho: float | None = None
        if len(pairs) >= 3:
            rho, _ = stats.spearmanr(grader_scores, human_scores)
            spearman_rho = float(rho)

        # Pass/fail agreement
        pass_agreements = [p.pass_agree for p in pairs]
        pass_fail_agreement = sum(pass_agreements) / len(pass_agreements)

        # Cohen's kappa
        cohens_kappa = self._compute_cohens_kappa(pairs)

        # Score deltas
        mean_score_delta = float(np.mean(deltas))
        mae = float(np.mean(np.abs(deltas)))
        grader_bias = mean_score_delta  # positive = grader scores higher

        # Calibration verdict
        is_calibrated = pearson_r is not None and pearson_r >= self.threshold

        return CalibrationResult(
            pairs=pairs,
            sample_count=len(pairs),
            pearson_r=pearson_r,
            spearman_rho=spearman_rho,
            pass_fail_agreement=pass_fail_agreement,
            cohens_kappa=cohens_kappa,
            mean_score_delta=mean_score_delta,
            mae=mae,
            grader_bias=grader_bias,
            threshold=self.threshold,
            is_calibrated=is_calibrated,
        )

    @staticmethod
    def _compute_cohens_kappa(pairs: list[CalibrationPair]) -> float | None:
        """Compute Cohen's kappa for pass/fail agreement."""
        if len(pairs) < 2:
            return None

        n = len(pairs)
        # Confusion matrix counts
        tp = sum(1 for p in pairs if p.grader_passed and p.human_passed)
        tn = sum(1 for p in pairs if not p.grader_passed and not p.human_passed)
        fp = sum(1 for p in pairs if p.grader_passed and not p.human_passed)
        fn = sum(1 for p in pairs if not p.grader_passed and p.human_passed)

        po = (tp + tn) / n  # Observed agreement
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)  # Expected

        if pe == 1.0:
            return 1.0

        return (po - pe) / (1.0 - pe)


@runtime_checkable
class OutcomeLike(Protocol):
    """Protocol for objects with .score and .passed attributes."""

    score: float
    passed: bool
