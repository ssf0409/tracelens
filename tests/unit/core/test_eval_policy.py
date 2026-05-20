"""Tests for three-way grader policy (EvalPolicy: gate/warn/track)."""

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


class TestEvalPolicy:
    """Tests for EvalPolicy enum."""

    def test_policy_values(self) -> None:
        assert EvalPolicy.GATE == "gate"
        assert EvalPolicy.WARN == "warn"
        assert EvalPolicy.TRACK == "track"

    def test_policy_is_str_enum(self) -> None:
        assert isinstance(EvalPolicy.GATE, str)


class TestGraderConfigPolicy:
    """Tests for policy field on GraderConfig."""

    def test_default_policy_is_track(self) -> None:
        config = GraderConfig()
        assert config.policy == EvalPolicy.TRACK

    def test_explicit_policy(self) -> None:
        config = GraderConfig(policy=EvalPolicy.GATE)
        assert config.policy == EvalPolicy.GATE

    def test_threshold_field(self) -> None:
        config = GraderConfig(policy=EvalPolicy.WARN, threshold=0.7)
        assert config.threshold == 0.7

    def test_threshold_default_none(self) -> None:
        config = GraderConfig()
        assert config.threshold is None


class TestBackwardCompatibility:
    """GraderRole must still work for existing code."""

    def test_grader_role_still_exists(self) -> None:
        assert GraderRole.MUST_PASS == "must_pass"
        assert GraderRole.SCORE_CONTRIBUTOR == "score_contributor"

    def test_grader_config_role_still_works(self) -> None:
        config = GraderConfig(role=GraderRole.MUST_PASS)
        assert config.role == GraderRole.MUST_PASS

    def test_is_must_pass_still_works(self) -> None:
        config = GraderConfig(role=GraderRole.MUST_PASS)
        grader = SimpleScoreGrader("test", score=0.8, passed=True, config=config)
        assert grader.is_must_pass is True


class TestGraderPolicyProperties:
    """Tests for policy-related properties on Grader."""

    def test_grader_policy_property(self) -> None:
        config = GraderConfig(policy=EvalPolicy.GATE)
        grader = SimpleScoreGrader("test", score=0.8, passed=True, config=config)
        assert grader.policy == EvalPolicy.GATE

    def test_is_gate(self) -> None:
        config = GraderConfig(policy=EvalPolicy.GATE)
        grader = SimpleScoreGrader("test", score=0.8, passed=True, config=config)
        assert grader.is_gate is True
        assert grader.is_warn is False
        assert grader.is_track is False

    def test_is_warn(self) -> None:
        config = GraderConfig(policy=EvalPolicy.WARN)
        grader = SimpleScoreGrader("test", score=0.8, passed=True, config=config)
        assert grader.is_warn is True

    def test_is_track(self) -> None:
        config = GraderConfig(policy=EvalPolicy.TRACK)
        grader = SimpleScoreGrader("test", score=0.8, passed=True, config=config)
        assert grader.is_track is True


class TestCompositeGraderPolicy:
    """Tests for CompositeGrader with policy-based aggregation."""

    @pytest.fixture
    def sample_task(self):
        return Task(task_id="test-task", name="Test", input_data={"test": "data"})

    @pytest.fixture
    def sample_transcript(self, sample_task):
        return Transcript(task_id=sample_task.task_id, final_output={"result": "test"})

    @pytest.mark.asyncio
    async def test_gate_failure_causes_overall_failure(
        self, sample_task, sample_transcript
    ) -> None:
        """Gate grader failure causes overall failure."""
        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "safety",
                    score=0.3,
                    passed=False,
                    config=GraderConfig(policy=EvalPolicy.GATE),
                ), 0.2),
                (SimpleScoreGrader(
                    "quality",
                    score=0.95,
                    passed=True,
                    config=GraderConfig(policy=EvalPolicy.TRACK),
                ), 0.8),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)
        assert outcome.passed is False

    @pytest.mark.asyncio
    async def test_warn_failure_doesnt_fail_overall(
        self, sample_task, sample_transcript
    ) -> None:
        """Warn grader failure doesn't fail overall (by default)."""
        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "latency",
                    score=0.3,
                    passed=False,
                    config=GraderConfig(policy=EvalPolicy.WARN),
                ), 0.5),
                (SimpleScoreGrader(
                    "quality",
                    score=0.9,
                    passed=True,
                    config=GraderConfig(policy=EvalPolicy.TRACK),
                ), 0.5),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)
        # Warn failures don't block by default
        assert outcome.passed is True

    @pytest.mark.asyncio
    async def test_track_failure_doesnt_fail_overall(
        self, sample_task, sample_transcript
    ) -> None:
        """Track grader failure never fails overall."""
        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "clarity",
                    score=0.2,
                    passed=False,
                    config=GraderConfig(policy=EvalPolicy.TRACK),
                ), 1.0),
            ],
        )

        outcome = await composite.grade(sample_transcript, sample_task)
        assert outcome.passed is True

    def test_graders_by_policy_property(self) -> None:
        """Test filtering graders by policy."""
        composite = CompositeGrader(
            grader_id="composite",
            graders=[
                (SimpleScoreGrader(
                    "safety", score=1.0, passed=True,
                    config=GraderConfig(policy=EvalPolicy.GATE),
                ), 0.3),
                (SimpleScoreGrader(
                    "latency", score=0.8, passed=True,
                    config=GraderConfig(policy=EvalPolicy.WARN),
                ), 0.3),
                (SimpleScoreGrader(
                    "quality", score=0.8, passed=True,
                    config=GraderConfig(policy=EvalPolicy.TRACK),
                ), 0.4),
            ],
        )

        assert len(composite.gate_graders) == 1
        assert len(composite.warn_graders) == 1
        assert len(composite.track_graders) == 1
