"""Grader abstractions for evaluating agent outputs.

Graders evaluate Transcripts and produce Outcomes. There are three main types:
- CodeGrader: Deterministic, code-based grading
- LLMGrader: LLM-as-judge grading
- HumanGrader: Human evaluation (for calibration)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel

from eval_kit.core.outcome import Outcome
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript


class GraderType(str, Enum):
    """Types of graders."""

    CODE_BASED = "code_based"      # Deterministic, code logic
    LLM_BASED = "llm_based"        # LLM-as-judge
    HUMAN = "human"                # Human evaluation
    COMPOSITE = "composite"        # Combination of multiple graders


class GraderRole(str, Enum):
    """Role of a grader in composite scoring.

    MUST_PASS: Safety/constraint graders. If ANY must-pass grader fails,
               the entire trial fails regardless of other scores.
               Example: safety checks, constraint validation, format validation

    SCORE_CONTRIBUTOR: Quality/style graders. Contribute to weighted average.
                       Failure reduces score but doesn't automatically fail trial.
                       Example: quality metrics, style, personalization
    """

    MUST_PASS = "must_pass"
    SCORE_CONTRIBUTOR = "score_contributor"


class GraderConfig(BaseModel):
    """Configuration for a grader."""

    pass_threshold: float = 0.5
    timeout_seconds: float = 60.0
    retry_on_error: bool = True
    max_retries: int = 3

    # For LLM graders
    model: str | None = None
    temperature: float = 0.0

    # For composite graders
    weight: float = 1.0

    # Role in composite grading
    role: GraderRole = GraderRole.SCORE_CONTRIBUTOR


class Grader(ABC):
    """Abstract base class for all graders.

    Graders evaluate agent outputs (Transcripts) and produce Outcomes.
    Subclass either CodeGrader or LLMGrader for specific implementations.

    Example:
        class MyGrader(CodeGrader):
            def compute_metrics(self, transcript, task):
                return {"accuracy": 0.95}

            def determine_pass(self, metrics, task):
                return metrics["accuracy"] >= 0.9, metrics["accuracy"]
    """

    def __init__(
        self,
        grader_id: str,
        config: GraderConfig | None = None,
    ):
        self.grader_id = grader_id
        self.config = config or GraderConfig()

    @property
    @abstractmethod
    def grader_type(self) -> GraderType:
        """Return the type of this grader."""
        pass

    @property
    def is_deterministic(self) -> bool:
        """Whether this grader produces deterministic results."""
        return self.grader_type == GraderType.CODE_BASED

    @property
    def requires_llm(self) -> bool:
        """Whether this grader requires LLM calls."""
        return self.grader_type == GraderType.LLM_BASED

    @property
    def requires_human(self) -> bool:
        """Whether this grader requires human input."""
        return self.grader_type == GraderType.HUMAN

    @property
    def role(self) -> GraderRole:
        """Role of this grader in composite scoring."""
        return self.config.role

    @property
    def is_must_pass(self) -> bool:
        """Whether this grader must pass for trial to pass."""
        return self.role == GraderRole.MUST_PASS

    @property
    def is_score_contributor(self) -> bool:
        """Whether this grader contributes to score average."""
        return self.role == GraderRole.SCORE_CONTRIBUTOR

    @abstractmethod
    async def grade(
        self,
        transcript: Transcript,
        task: Task,
    ) -> Outcome:
        """Grade the transcript for the given task.

        Args:
            transcript: The agent's execution record
            task: The task being evaluated

        Returns:
            An Outcome with pass/fail, score, and metrics
        """
        pass

    def create_outcome(
        self,
        trial_id: str,
        passed: bool,
        score: float,
        metrics: dict[str, float] | None = None,
        feedback: str | None = None,
        **kwargs: Any,
    ) -> Outcome:
        """Helper to create an Outcome with common fields."""
        return Outcome(
            outcome_id=str(uuid.uuid4()),
            trial_id=trial_id,
            grader_id=self.grader_id,
            passed=passed,
            score=score,
            metrics=metrics or {},
            feedback=feedback,
            **kwargs,
        )


class CodeGrader(Grader):
    """Base class for deterministic code-based graders.

    CodeGraders compute metrics from the transcript and then determine
    pass/fail based on those metrics. They are deterministic - the same
    input always produces the same output.

    Use for: objective metrics (Sharpe ratio, accuracy, latency)

    Example:
        class FinancialGrader(CodeGrader):
            def compute_metrics(self, transcript, task):
                returns = transcript.final_output["returns"]
                return {
                    "sharpe_ratio": calculate_sharpe(returns),
                    "max_drawdown": calculate_max_dd(returns),
                }

            def determine_pass(self, metrics, task):
                passed = metrics["sharpe_ratio"] >= 1.0
                score = min(metrics["sharpe_ratio"] / 2.0, 1.0)
                return passed, score
    """

    @property
    def grader_type(self) -> GraderType:
        return GraderType.CODE_BASED

    @abstractmethod
    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        """Compute grading metrics from transcript.

        Implement this to extract and calculate metrics from the
        agent's execution record.
        """
        pass

    @abstractmethod
    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        """Determine pass/fail and score from metrics.

        Returns:
            Tuple of (passed, score) where score is 0-1 normalized.
        """
        pass

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        """Grade by computing metrics and determining pass/fail."""
        metrics = self.compute_metrics(transcript, task)
        passed, score = self.determine_pass(metrics, task)

        return self.create_outcome(
            trial_id=transcript.task_id,  # Will be updated by runner
            passed=passed,
            score=score,
            metrics=metrics,
        )


class LLMGrader(Grader):
    """Base class for LLM-as-judge graders.

    LLMGraders use an LLM to evaluate agent outputs. They are non-deterministic
    and require careful prompt engineering.

    Use for: subjective quality (specificity, personalization, clarity)

    Example:
        class QualityGrader(LLMGrader):
            def build_grading_prompt(self, transcript, task):
                return f'''Evaluate the quality of this output:
                {transcript.final_output}

                Score 1-10 on: clarity, completeness, accuracy
                Return JSON: {{"score": X, "feedback": "..."}}'''

            def parse_llm_response(self, response, task):
                data = json.loads(response)
                score = data["score"] / 10.0
                passed = score >= 0.7
                return passed, score, {}, data["feedback"]
    """

    def __init__(
        self,
        grader_id: str,
        model: str = "gpt-4",
        config: GraderConfig | None = None,
    ):
        super().__init__(grader_id, config)
        self.model = model

    @property
    def grader_type(self) -> GraderType:
        return GraderType.LLM_BASED

    @abstractmethod
    def build_grading_prompt(
        self,
        transcript: Transcript,
        task: Task,
    ) -> str:
        """Build the prompt for LLM grading.

        Implement this to create a prompt that instructs the LLM
        how to evaluate the agent's output.
        """
        pass

    @abstractmethod
    def parse_llm_response(
        self,
        response: str,
        task: Task,
    ) -> tuple[bool, float, dict[str, float], str]:
        """Parse LLM response into structured result.

        Returns:
            Tuple of (passed, score, metrics, feedback)
        """
        pass

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM. Override this to use your preferred client.

        Default implementation raises NotImplementedError.
        In practice, integrate with OpenAI, Anthropic, or LiteLLM.
        """
        raise NotImplementedError(
            "Subclass must implement _call_llm or use a mixin"
        )

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        """Grade by calling LLM and parsing response."""
        prompt = self.build_grading_prompt(transcript, task)

        response = await self._call_llm(prompt)

        passed, score, metrics, feedback = self.parse_llm_response(response, task)

        return self.create_outcome(
            trial_id=transcript.task_id,
            passed=passed,
            score=score,
            metrics=metrics,
            feedback=feedback,
            reasoning=response,  # Store raw LLM response
        )


class CompositeGrader(Grader):
    """Combines multiple graders with role-based aggregation.

    Supports two types of graders:
    - MUST_PASS: Any failure causes entire trial to fail (safety, constraints)
    - SCORE_CONTRIBUTOR: Contributes to weighted score average (quality, style)

    The overall trial passes only if ALL must-pass graders pass.
    The score is a weighted average of all graders (must-pass get full score if pass).

    Example:
        # Safety grader (must pass)
        safety_config = GraderConfig(role=GraderRole.MUST_PASS)
        safety_grader = SafetyGrader("safety", config=safety_config)

        # Quality graders (contribute to score)
        quality_config = GraderConfig(role=GraderRole.SCORE_CONTRIBUTOR)
        quality_grader = QualityGrader("quality", config=quality_config)

        composite = CompositeGrader(
            grader_id="combined",
            graders=[
                (safety_grader, 0.2),     # Weight for must-pass still affects score
                (quality_grader, 0.8),    # Higher weight for quality
            ],
        )
    """

    def __init__(
        self,
        grader_id: str,
        graders: list[tuple[Grader, float]],  # (grader, weight)
        config: GraderConfig | None = None,
    ):
        super().__init__(grader_id, config)
        self.graders = graders
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Ensure weights sum to 1.0."""
        total = sum(w for _, w in self.graders)
        if total > 0:
            self.graders = [(g, w / total) for g, w in self.graders]

    @property
    def grader_type(self) -> GraderType:
        return GraderType.COMPOSITE

    @property
    def must_pass_graders(self) -> list[tuple[Grader, float]]:
        """Get all must-pass graders."""
        return [(g, w) for g, w in self.graders if g.is_must_pass]

    @property
    def score_contributor_graders(self) -> list[tuple[Grader, float]]:
        """Get all score-contributor graders."""
        return [(g, w) for g, w in self.graders if g.is_score_contributor]

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        """Grade using role-based aggregation.

        1. Run all must-pass graders - any failure causes trial to fail
        2. Run all score-contributor graders
        3. Compute weighted score from all graders
        4. Overall pass requires all must-pass graders to pass
        """
        all_metrics: dict[str, float] = {}
        total_score = 0.0
        feedbacks: list[str] = []

        # Track must-pass results
        must_pass_results: list[tuple[Grader, Outcome]] = []
        score_contributor_results: list[tuple[Grader, float, Outcome]] = []

        # Run all graders and collect results
        for grader, weight in self.graders:
            outcome = await grader.grade(transcript, task)

            # Prefix metrics with grader ID
            for metric, value in outcome.metrics.items():
                all_metrics[f"{grader.grader_id}.{metric}"] = value

            # Compute weighted score contribution
            total_score += outcome.score * weight

            if outcome.feedback:
                role_prefix = "[MUST-PASS]" if grader.is_must_pass else ""
                feedbacks.append(
                    f"{role_prefix}[{grader.grader_id}] {outcome.feedback}"
                )

            # Track by role
            if grader.is_must_pass:
                must_pass_results.append((grader, outcome))
            else:
                score_contributor_results.append((grader, weight, outcome))

        # Determine overall pass/fail
        # ALL must-pass graders must pass for trial to pass
        must_pass_all_passed = all(
            outcome.passed for _, outcome in must_pass_results
        )

        # Add metadata about must-pass failures
        failed_must_pass = [
            grader.grader_id
            for grader, outcome in must_pass_results
            if not outcome.passed
        ]

        if failed_must_pass:
            all_metrics["_failed_must_pass"] = len(failed_must_pass)
            feedbacks.insert(
                0,
                f"MUST-PASS FAILURE: {', '.join(failed_must_pass)}"
            )

        # Overall pass requires all must-pass to pass
        overall_passed = must_pass_all_passed

        return self.create_outcome(
            trial_id=transcript.task_id,
            passed=overall_passed,
            score=total_score,
            metrics=all_metrics,
            feedback="\n".join(feedbacks) if feedbacks else None,
        )
