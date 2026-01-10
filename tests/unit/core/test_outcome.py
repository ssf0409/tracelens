"""Tests for outcome module."""

import pytest

from agent_eval.core.outcome import Outcome, GradeLevel, AggregatedOutcome


class TestGradeLevel:
    """Tests for GradeLevel enum."""

    def test_from_score_excellent(self):
        """Test excellent grade."""
        assert GradeLevel.from_score(0.95) == GradeLevel.EXCELLENT
        assert GradeLevel.from_score(0.90) == GradeLevel.EXCELLENT

    def test_from_score_good(self):
        """Test good grade."""
        assert GradeLevel.from_score(0.85) == GradeLevel.GOOD
        assert GradeLevel.from_score(0.70) == GradeLevel.GOOD

    def test_from_score_acceptable(self):
        """Test acceptable grade."""
        assert GradeLevel.from_score(0.65) == GradeLevel.ACCEPTABLE
        assert GradeLevel.from_score(0.50) == GradeLevel.ACCEPTABLE

    def test_from_score_poor(self):
        """Test poor grade."""
        assert GradeLevel.from_score(0.40) == GradeLevel.POOR
        assert GradeLevel.from_score(0.30) == GradeLevel.POOR

    def test_from_score_fail(self):
        """Test fail grade."""
        assert GradeLevel.from_score(0.25) == GradeLevel.FAIL
        assert GradeLevel.from_score(0.0) == GradeLevel.FAIL


class TestOutcome:
    """Tests for Outcome model."""

    def test_outcome_creation(self, sample_outcome: Outcome):
        """Test creating an outcome."""
        assert sample_outcome.passed is True
        assert sample_outcome.score == 0.85
        assert sample_outcome.grade_level == GradeLevel.GOOD
        assert "specificity" in sample_outcome.metrics

    def test_outcome_grade_level_auto_set(self):
        """Test grade level is auto-set from score."""
        outcome = Outcome(
            trial_id="test",
            grader_id="test",
            passed=True,
            score=0.95,
        )

        assert outcome.grade_level == GradeLevel.EXCELLENT

    def test_outcome_to_summary_dict(self, sample_outcome: Outcome):
        """Test outcome summary."""
        summary = sample_outcome.to_summary_dict()

        assert summary["passed"] is True
        assert summary["score"] == 0.85
        assert summary["grade_level"] == "good"
        assert "metrics" in summary

    def test_outcome_to_ci_dict(self, sample_outcome: Outcome):
        """Test CI output format."""
        ci_dict = sample_outcome.to_ci_dict()

        assert ci_dict["grader"] == "quality_grader"
        assert ci_dict["passed"] is True
        assert ci_dict["score"] == 0.85

    def test_outcome_with_confidence(self):
        """Test outcome with confidence score."""
        outcome = Outcome(
            trial_id="test",
            grader_id="llm_grader",
            passed=True,
            score=0.8,
            confidence=0.95,
        )

        assert outcome.confidence == 0.95


class TestAggregatedOutcome:
    """Tests for AggregatedOutcome model."""

    def test_from_outcomes_empty(self):
        """Test aggregation of empty list."""
        agg = AggregatedOutcome.from_outcomes([])

        assert agg.total_trials == 0
        assert agg.pass_rate == 0.0

    def test_from_outcomes(self):
        """Test aggregation of outcomes."""
        outcomes = [
            Outcome(trial_id="t1", grader_id="g1", passed=True, score=0.9,
                    metrics={"quality": 0.9}),
            Outcome(trial_id="t2", grader_id="g1", passed=True, score=0.8,
                    metrics={"quality": 0.8}),
            Outcome(trial_id="t3", grader_id="g1", passed=False, score=0.4,
                    metrics={"quality": 0.4}),
        ]

        agg = AggregatedOutcome.from_outcomes(outcomes)

        assert agg.total_trials == 3
        assert agg.passed_trials == 2
        assert agg.failed_trials == 1
        assert agg.pass_rate == pytest.approx(2 / 3, rel=0.01)
        assert agg.mean_score == pytest.approx(0.7, rel=0.01)
        assert agg.min_score == 0.4
        assert agg.max_score == 0.9

    def test_from_outcomes_multiple_graders(self):
        """Test aggregation with multiple graders."""
        outcomes = [
            Outcome(trial_id="t1", grader_id="g1", passed=True, score=0.9),
            Outcome(trial_id="t1", grader_id="g2", passed=True, score=0.8),
            Outcome(trial_id="t2", grader_id="g1", passed=False, score=0.4),
            Outcome(trial_id="t2", grader_id="g2", passed=True, score=0.7),
        ]

        agg = AggregatedOutcome.from_outcomes(outcomes)

        assert agg.total_trials == 4
        assert "g1" in agg.grader_pass_rates
        assert "g2" in agg.grader_pass_rates
        assert agg.grader_pass_rates["g1"] == 0.5  # 1/2
        assert agg.grader_pass_rates["g2"] == 1.0  # 2/2
