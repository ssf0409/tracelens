"""Tests for CalibrationAnalyzer."""

import pytest

from tracelens.calibration.analyzer import (
    AnnotationSet,
    CalibrationAnalyzer,
    HumanAnnotation,
)
from tracelens.core.outcome import Outcome


def _make_outcome(score: float, passed: bool) -> Outcome:
    return Outcome(
        trial_id="trial",
        grader_id="grader",
        passed=passed,
        score=score,
    )


class TestAnnotationSet:
    def test_from_json_list(self):
        data = [
            {"task_id": "t1", "human_score": 0.8, "human_passed": True, "notes": "good"},
            {"task_id": "t2", "human_score": 0.3, "human_passed": False},
        ]
        annotations = AnnotationSet.from_json_list(data)
        assert len(annotations.annotations) == 2
        assert annotations.annotations[0].task_id == "t1"

    def test_get_by_task_id(self):
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.8, human_passed=True),
        ])
        assert annotations.get_by_task_id("t1") is not None
        assert annotations.get_by_task_id("missing") is None


class TestCalibrationAnalyzer:
    def test_perfect_agreement(self):
        """Grader and human agree perfectly → is_calibrated=True."""
        outcomes = {
            "t1": _make_outcome(0.9, True),
            "t2": _make_outcome(0.7, True),
            "t3": _make_outcome(0.3, False),
            "t4": _make_outcome(0.1, False),
        }
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.9, human_passed=True),
            HumanAnnotation(task_id="t2", human_score=0.7, human_passed=True),
            HumanAnnotation(task_id="t3", human_score=0.3, human_passed=False),
            HumanAnnotation(task_id="t4", human_score=0.1, human_passed=False),
        ])

        analyzer = CalibrationAnalyzer(threshold=0.7)
        result = analyzer.analyze(outcomes, annotations)

        assert result.is_calibrated is True
        assert result.pearson_r == pytest.approx(1.0)
        assert result.pass_fail_agreement == 1.0
        assert result.cohens_kappa == pytest.approx(1.0)
        assert result.mae == pytest.approx(0.0)

    def test_no_agreement(self):
        """Grader scores inversely to human → low correlation."""
        outcomes = {
            "t1": _make_outcome(0.1, False),
            "t2": _make_outcome(0.3, False),
            "t3": _make_outcome(0.7, True),
            "t4": _make_outcome(0.9, True),
        }
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.9, human_passed=True),
            HumanAnnotation(task_id="t2", human_score=0.7, human_passed=True),
            HumanAnnotation(task_id="t3", human_score=0.3, human_passed=False),
            HumanAnnotation(task_id="t4", human_score=0.1, human_passed=False),
        ])

        analyzer = CalibrationAnalyzer(threshold=0.7)
        result = analyzer.analyze(outcomes, annotations)

        assert result.is_calibrated is False
        assert result.pearson_r is not None
        assert result.pearson_r < 0  # Negative correlation

    def test_positive_bias(self):
        """Grader consistently scores higher than humans → positive bias."""
        outcomes = {
            "t1": _make_outcome(0.9, True),
            "t2": _make_outcome(0.8, True),
            "t3": _make_outcome(0.6, True),
        }
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.7, human_passed=True),
            HumanAnnotation(task_id="t2", human_score=0.6, human_passed=True),
            HumanAnnotation(task_id="t3", human_score=0.4, human_passed=False),
        ])

        analyzer = CalibrationAnalyzer(threshold=0.7)
        result = analyzer.analyze(outcomes, annotations)

        assert result.grader_bias is not None
        assert result.grader_bias > 0  # Positive bias

    def test_threshold_behavior(self):
        """Threshold determines is_calibrated cutoff."""
        # Correlated but not perfectly
        outcomes = {
            "t1": _make_outcome(0.85, True),
            "t2": _make_outcome(0.65, True),
            "t3": _make_outcome(0.35, False),
            "t4": _make_outcome(0.15, False),
        }
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.9, human_passed=True),
            HumanAnnotation(task_id="t2", human_score=0.7, human_passed=True),
            HumanAnnotation(task_id="t3", human_score=0.3, human_passed=False),
            HumanAnnotation(task_id="t4", human_score=0.1, human_passed=False),
        ])

        # Very high threshold
        strict = CalibrationAnalyzer(threshold=0.999)
        result_strict = strict.analyze(outcomes, annotations)

        # Low threshold
        loose = CalibrationAnalyzer(threshold=0.5)
        result_loose = loose.analyze(outcomes, annotations)

        # Same pearson_r, different verdicts
        assert result_strict.pearson_r == result_loose.pearson_r
        # Strict may fail, loose should pass (high correlation data)
        assert result_loose.is_calibrated is True

    def test_too_few_samples(self):
        """With < 2 samples, result is not calibrated."""
        outcomes = {"t1": _make_outcome(0.9, True)}
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.9, human_passed=True),
        ])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze(outcomes, annotations)

        assert result.is_calibrated is False
        assert result.sample_count == 1

    def test_cohens_kappa_partial(self):
        """Cohen's kappa with some disagreement."""
        outcomes = {
            "t1": _make_outcome(0.8, True),
            "t2": _make_outcome(0.6, True),   # Grader says pass
            "t3": _make_outcome(0.3, False),
            "t4": _make_outcome(0.1, False),
        }
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.8, human_passed=True),
            HumanAnnotation(task_id="t2", human_score=0.4, human_passed=False),  # Human says fail
            HumanAnnotation(task_id="t3", human_score=0.3, human_passed=False),
            HumanAnnotation(task_id="t4", human_score=0.1, human_passed=False),
        ])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze(outcomes, annotations)

        assert result.cohens_kappa is not None
        assert result.cohens_kappa < 1.0  # Not perfect
        assert result.pass_fail_agreement == 0.75  # 3/4 agree

    def test_to_dict(self):
        outcomes = {
            "t1": _make_outcome(0.9, True),
            "t2": _make_outcome(0.3, False),
        }
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.9, human_passed=True),
            HumanAnnotation(task_id="t2", human_score=0.3, human_passed=False),
        ])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze(outcomes, annotations)
        d = result.to_dict()

        assert "pearson_r" in d
        assert "is_calibrated" in d
        assert "threshold" in d

    def test_render_table(self):
        outcomes = {
            "t1": _make_outcome(0.9, True),
            "t2": _make_outcome(0.3, False),
        }
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.9, human_passed=True),
            HumanAnnotation(task_id="t2", human_score=0.3, human_passed=False),
        ])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze(outcomes, annotations)
        table = result.render_table()

        assert "Calibration Report" in table
        assert "Pearson r" in table

    def test_missing_outcome_skipped(self):
        """Annotations without matching outcomes are skipped."""
        outcomes = {"t1": _make_outcome(0.9, True)}
        annotations = AnnotationSet(annotations=[
            HumanAnnotation(task_id="t1", human_score=0.9, human_passed=True),
            HumanAnnotation(task_id="t_missing", human_score=0.5, human_passed=True),
        ])

        analyzer = CalibrationAnalyzer()
        result = analyzer.analyze(outcomes, annotations)
        assert result.sample_count == 1
