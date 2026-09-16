"""Tests for grader role functionality (must-pass vs score-contributor)."""

import pytest

from tracelens.core.grader import (
    CodeGrader,
    CompositeGrader,
    EvalPolicy,
    GraderConfig,
    GraderRole,
)
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript


class SimpleScoreGrader(CodeGrader):
    """Test grader that returns a fixed score."""

    def __init__(
        self,
        grader_id: str,
        score: float,
        passed: bool,
        config: GraderConfig | None = None,
    ):
        super().__init__(grader_id, config)
        self._score = score
        self._passed = passed

    def compute_metrics(self, transcript, task):
        return {"score": self._score}

    def determine_pass(self, metrics, task):
        return self._passed, self._score


class TestGraderRole:
    """Tests for GraderRole enum."""

    def test_role_values(self):
        """Test role enum values."""
        assert GraderRole.MUST_PASS == "must_pass"
        assert GraderRole.SCORE_CONTRIBUTOR == "score_contributor"


class TestGraderRoleProperties:
    """Tests for grader role properties."""

    def test_default_role_is_score_contributor(self):
        """Test that default role is score-contributor."""
        grader = SimpleScoreGrader("test", score=0.8, passed=True)
        assert grader.role == GraderRole.SCORE_CONTRIBUTOR
        assert grader.is_score_contributor is True
        assert grader.is_must_pass is False

    def test_must_pass_role(self):
        """Test setting must-pass role."""
        config = GraderConfig(role=GraderRole.MUST_PASS)
        grader = SimpleScoreGrader("test", score=0.8, passed=True, config=config)
        assert grader.role == GraderRole.MUST_PASS
        assert grader.is_must_pass is True
        assert grader.is_score_contributor is False


class TestCompositeGraderRoles:
    """Tests for CompositeGrader with roles."""

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return Task(
            task_id="test-task",
            name="Test Task",
            input_data={"test": "data"},
        )

    @pytest.fixture
    def sample_transcript(self, sample_task):
        """Create a sample transcript."""
        return Transcript(
            task_id=sample_task.task_id,
            final_output={"result": "test"},
        )

    @pytest.mark.asyncio
    async def test_all_score_contributors_pass(self, sample_task, sample_transcript):
        """Test composite with all score-contributors passing."""
        config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader("g1", score=0.8, passed=True, config=config), 0.5),
                (SimpleScoreGrader("g2", score=0.9, passed=True, config=config), 0.5),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)

        assert outcome.passed is True
        assert outcome.score == pytest.approx(0.85, rel=0.01)

    @pytest.mark.asyncio
    async def test_must_pass_failure_causes_overall_failure(
        self, sample_task, sample_transcript
    ):
        """Test that must-pass failure causes overall failure."""
        must_pass_config = GraderConfig(role=GraderRole.MUST_PASS)
        score_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                # Must-pass grader fails
                (SimpleScoreGrader(
                    "safety", score=0.3, passed=False, config=must_pass_config
                ), 0.2),
                # Score contributor passes with high score
                (SimpleScoreGrader(
                    "quality", score=0.95, passed=True, config=score_config
                ), 0.8),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)

        # Overall should fail despite high quality score
        assert outcome.passed is False
        # Score is still computed
        assert outcome.score > 0.5  # Weighted average

    @pytest.mark.asyncio
    async def test_must_pass_success_allows_overall_pass(
        self, sample_task, sample_transcript
    ):
        """Test that must-pass success allows overall pass."""
        must_pass_config = GraderConfig(role=GraderRole.MUST_PASS)
        score_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                # Must-pass grader passes
                (SimpleScoreGrader(
                    "safety", score=1.0, passed=True, config=must_pass_config
                ), 0.3),
                # Score contributor also passes
                (SimpleScoreGrader(
                    "quality", score=0.8, passed=True, config=score_config
                ), 0.7),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)

        assert outcome.passed is True
        # Weighted average: 1.0*0.3 + 0.8*0.7 = 0.3 + 0.56 = 0.86
        assert outcome.score == pytest.approx(0.86, rel=0.01)

    @pytest.mark.asyncio
    async def test_multiple_must_pass_all_must_pass(
        self, sample_task, sample_transcript
    ):
        """Test that ALL must-pass graders must pass."""
        must_pass_config = GraderConfig(role=GraderRole.MUST_PASS)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "safety1", score=1.0, passed=True, config=must_pass_config
                ), 0.5),
                # Second must-pass fails
                (SimpleScoreGrader(
                    "safety2", score=0.2, passed=False, config=must_pass_config
                ), 0.5),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)

        # Should fail because one must-pass failed
        assert outcome.passed is False

    @pytest.mark.asyncio
    async def test_failed_must_pass_tracked_in_metrics(
        self, sample_task, sample_transcript
    ):
        """Test that failed must-pass graders are tracked."""
        must_pass_config = GraderConfig(role=GraderRole.MUST_PASS)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "safety", score=0.0, passed=False, config=must_pass_config
                ), 1.0),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)

        assert "_failed_must_pass" in outcome.metrics
        assert outcome.metrics["_failed_must_pass"] == 1
        assert "MUST-PASS FAILURE" in outcome.feedback

    @pytest.mark.asyncio
    async def test_score_contributor_failure_doesnt_fail_overall(
        self, sample_task, sample_transcript
    ):
        """Test that score-contributor failure doesn't fail overall."""
        must_pass_config = GraderConfig(role=GraderRole.MUST_PASS)
        score_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                # Must-pass passes
                (SimpleScoreGrader(
                    "safety", score=1.0, passed=True, config=must_pass_config
                ), 0.3),
                # Score contributor fails (low score)
                (SimpleScoreGrader(
                    "quality", score=0.3, passed=False, config=score_config
                ), 0.7),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)

        # Should pass because must-pass passed
        # (score-contributor failure just reduces score)
        assert outcome.passed is True
        # Score is weighted average: 1.0*0.3 + 0.3*0.7 = 0.51
        assert outcome.score == pytest.approx(0.51, rel=0.01)

    def test_must_pass_graders_property(self):
        """Test must_pass_graders property."""
        must_pass_config = GraderConfig(role=GraderRole.MUST_PASS)
        score_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "safety", score=1.0, passed=True, config=must_pass_config
                ), 0.3),
                (SimpleScoreGrader(
                    "quality", score=0.8, passed=True, config=score_config
                ), 0.7),
            ],
        )

        assert len(composite.must_pass_graders) == 1
        assert composite.must_pass_graders[0][0].grader_id == "safety"

    def test_score_contributor_graders_property(self):
        """Test score_contributor_graders property."""
        must_pass_config = GraderConfig(role=GraderRole.MUST_PASS)
        score_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)

        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "safety", score=1.0, passed=True, config=must_pass_config
                ), 0.3),
                (SimpleScoreGrader(
                    "quality", score=0.8, passed=True, config=score_config
                ), 0.7),
            ],
        )

        assert len(composite.score_contributor_graders) == 1
        assert composite.score_contributor_graders[0][0].grader_id == "quality"


class TestGraderConfigRole:
    """Tests for GraderConfig role field."""

    def test_default_role(self):
        """Test default role in config."""
        config = GraderConfig()
        assert config.role == GraderRole.SCORE_CONTRIBUTOR

    def test_explicit_role(self):
        """Test explicit role setting."""
        config = GraderConfig(role=GraderRole.MUST_PASS)
        assert config.role == GraderRole.MUST_PASS


class TestCompositeGraderValidationAndFallback:
    """Tests for CompositeGrader construction validation and non-gated fallback."""

    def test_empty_graders_raises_value_error(self):
        with pytest.raises(ValueError, match="requires at least one sub-grader"):
            CompositeGrader("empty", graders=[])

    @pytest.mark.asyncio
    async def test_composite_without_blocking_graders_fails_if_any_sub_grader_fails(
        self, sample_task, sample_transcript
    ):
        """When no sub-grader is GATE or MUST_PASS, all sub-graders must pass."""
        g1 = SimpleScoreGrader("a", score=0.0, passed=False, config=GraderConfig(policy=EvalPolicy.TRACK))
        g2 = SimpleScoreGrader("b", score=0.0, passed=False, config=GraderConfig(policy=EvalPolicy.TRACK))

        composite = CompositeGrader("comp", graders=[(g1, 1.0), (g2, 1.0)])
        outcome = await composite.grade(sample_transcript, sample_task)

        assert outcome.passed is False
        assert outcome.score == 0.0
        assert "FAILURE (no gate configured, fallback to all): a, b" in outcome.feedback

    @pytest.mark.asyncio
    async def test_composite_without_blocking_graders_passes_if_all_pass(
        self, sample_task, sample_transcript
    ):
        """When no sub-grader is GATE/MUST_PASS, passes if all sub-graders pass."""
        g1 = SimpleScoreGrader("a", score=1.0, passed=True, config=GraderConfig(policy=EvalPolicy.TRACK))
        g2 = SimpleScoreGrader("b", score=1.0, passed=True, config=GraderConfig(policy=EvalPolicy.TRACK))

        composite = CompositeGrader("comp", graders=[(g1, 1.0), (g2, 1.0)])
        outcome = await composite.grade(sample_transcript, sample_task)

        assert outcome.passed is True
        assert outcome.score == 1.0

    @pytest.mark.asyncio
    async def test_composite_with_warn_only_fails_if_warn_fails(
        self, sample_task, sample_transcript
    ):
        """When WARN graders fail and no GATE is present, fallback causes failure."""
        g1 = SimpleScoreGrader("w", score=0.0, passed=False, config=GraderConfig(policy=EvalPolicy.WARN))
        g2 = SimpleScoreGrader("t", score=1.0, passed=True, config=GraderConfig(policy=EvalPolicy.TRACK))

        composite = CompositeGrader("comp", graders=[(g1, 1.0), (g2, 1.0)])
        outcome = await composite.grade(sample_transcript, sample_task)

        assert outcome.passed is False
        assert "FAILURE (no gate configured, fallback to all): w" in outcome.feedback
        assert "WARNING: w" in outcome.feedback
