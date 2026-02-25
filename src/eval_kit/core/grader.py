"""Grader abstractions for evaluating agent outputs.

Graders evaluate Transcripts and produce Outcomes. There are three main types:
- CodeGrader: Deterministic, code-based grading
- LLMGrader: LLM-as-judge grading
- HumanGrader: Human evaluation (for calibration)
"""

from __future__ import annotations

import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from eval_kit.core.outcome import Outcome
from eval_kit.core.task import Task
from eval_kit.core.transcript import Transcript

if TYPE_CHECKING:
    from eval_kit.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


class GraderType(str, Enum):
    """Types of graders."""

    CODE_BASED = "code_based"      # Deterministic, code logic
    LLM_BASED = "llm_based"        # LLM-as-judge
    HUMAN = "human"                # Human evaluation
    COMPOSITE = "composite"        # Combination of multiple graders


class EvalPolicy(str, Enum):
    """Three-way policy for how a grader's result affects CI.

    GATE:  Fails CI on violation. Use for hard constraints (valid JSON, no PII).
    WARN:  Warning by default, configurable CI fail. Use for regressions.
    TRACK: Never fails CI, just produces signals. Use for quality tracking.
    """

    GATE = "gate"
    WARN = "warn"
    TRACK = "track"


class GraderRole(str, Enum):
    """Deprecated: use EvalPolicy instead.

    Kept for backward compatibility. MUST_PASS maps to GATE, SCORE_CONTRIBUTOR maps to TRACK.
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

    # Three-way eval policy (replaces role for new code)
    policy: EvalPolicy = EvalPolicy.TRACK
    threshold: float | None = None

    # Deprecated: role in composite grading (kept for backward compat)
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

    @property
    def policy(self) -> EvalPolicy:
        """Three-way eval policy for this grader."""
        return self.config.policy

    @property
    def is_gate(self) -> bool:
        """Whether this grader is a gate (fails CI on violation)."""
        return self.policy == EvalPolicy.GATE

    @property
    def is_warn(self) -> bool:
        """Whether this grader is a warning."""
        return self.policy == EvalPolicy.WARN

    @property
    def is_track(self) -> bool:
        """Whether this grader is tracking-only."""
        return self.policy == EvalPolicy.TRACK

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
        provider: LLMProvider | None = None,
    ):
        super().__init__(grader_id, config)
        self.model = model
        self._provider = provider

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
        """Call the LLM via injected provider, or raise if none available.

        Override this in subclasses for custom LLM integration, or pass
        a ``provider`` to ``__init__`` to use the provider abstraction.
        """
        if self._provider is not None:
            return await self._provider.complete(prompt)
        raise NotImplementedError(
            "Pass a provider to __init__ or override _call_llm in a subclass"
        )

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        """Grade by calling LLM and parsing response."""
        prompt = self.build_grading_prompt(transcript, task)

        response = await self._call_llm(prompt)

        try:
            passed, score, metrics, feedback = self.parse_llm_response(response, task)
        except Exception as exc:
            preview = response[:200]
            raise RuntimeError(
                f"parse_llm_response failed for grader '{self.grader_id}': {exc}. "
                f"LLM response preview: {preview!r}"
            ) from exc

        return self.create_outcome(
            trial_id=transcript.task_id,
            passed=passed,
            score=score,
            metrics=metrics,
            feedback=feedback,
            reasoning=response,  # Store raw LLM response
        )


class CompositeGrader(Grader):
    """Combines multiple graders with policy-aware aggregation.

    Supports three policy tiers:
    - GATE: Any failure causes entire trial to fail (safety, constraints)
    - WARN: Failures emit warnings but don't fail by default
    - TRACK: Pure signals, never affect pass/fail

    Also supports legacy GraderRole (MUST_PASS maps to GATE behavior).

    The score is always a weighted average of all graders regardless of policy.
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

    # Legacy properties (backward compat)

    @property
    def must_pass_graders(self) -> list[tuple[Grader, float]]:
        """Get all must-pass graders (legacy + gate)."""
        return [(g, w) for g, w in self.graders if g.is_must_pass or g.is_gate]

    @property
    def score_contributor_graders(self) -> list[tuple[Grader, float]]:
        """Get all score-contributor graders (legacy)."""
        return [(g, w) for g, w in self.graders if g.is_score_contributor]

    # Policy-based properties

    @property
    def gate_graders(self) -> list[tuple[Grader, float]]:
        """Get all gate graders."""
        return [(g, w) for g, w in self.graders if g.is_gate]

    @property
    def warn_graders(self) -> list[tuple[Grader, float]]:
        """Get all warn graders."""
        return [(g, w) for g, w in self.graders if g.is_warn]

    @property
    def track_graders(self) -> list[tuple[Grader, float]]:
        """Get all track graders."""
        return [(g, w) for g, w in self.graders if g.is_track]

    def _is_blocking_grader(self, grader: Grader) -> bool:
        """Check if a grader failure should block the overall result.

        GATE policy or legacy MUST_PASS role → blocking.
        """
        return grader.is_gate or grader.is_must_pass

    async def grade(self, transcript: Transcript, task: Task) -> Outcome:
        """Grade using policy-aware aggregation.

        1. Run all graders and collect outcomes
        2. Compute weighted score from all graders
        3. Only GATE (or legacy MUST_PASS) failures cause overall failure
        4. WARN failures are recorded in feedback
        5. TRACK results are pure signals
        """
        all_metrics: dict[str, float] = {}
        total_score = 0.0
        feedbacks: list[str] = []

        blocking_results: list[tuple[Grader, Outcome]] = []
        warn_results: list[tuple[Grader, Outcome]] = []

        for grader, weight in self.graders:
            try:
                outcome = await grader.grade(transcript, task)
            except Exception as exc:
                logger.error(
                    "Sub-grader '%s' crashed: %s\n%s",
                    grader.grader_id,
                    exc,
                    traceback.format_exc(),
                )
                outcome = self.create_outcome(
                    trial_id=transcript.task_id,
                    passed=False,
                    score=0.0,
                    feedback=f"Sub-grader '{grader.grader_id}' crashed: {exc}",
                    metrics={"_grader_error": 1.0},
                )

            # Prefix metrics with grader ID
            for metric, value in outcome.metrics.items():
                all_metrics[f"{grader.grader_id}.{metric}"] = value

            total_score += outcome.score * weight

            if outcome.feedback:
                is_policy_tagged = grader.is_gate or grader.is_warn
                policy_tag = f"[{grader.policy.value.upper()}]" if is_policy_tagged else ""
                legacy_tag = "[MUST-PASS]" if grader.is_must_pass and not grader.is_gate else ""
                prefix = policy_tag or legacy_tag
                feedbacks.append(f"{prefix}[{grader.grader_id}] {outcome.feedback}")

            if self._is_blocking_grader(grader):
                blocking_results.append((grader, outcome))
            elif grader.is_warn:
                warn_results.append((grader, outcome))

        # GATE/MUST_PASS failures block
        all_blocking_passed = all(
            outcome.passed for _, outcome in blocking_results
        )
        failed_blocking = [
            grader.grader_id
            for grader, outcome in blocking_results
            if not outcome.passed
        ]
        if failed_blocking:
            all_metrics["_failed_must_pass"] = len(failed_blocking)
            feedbacks.insert(0, f"MUST-PASS FAILURE: {', '.join(failed_blocking)}")

        # WARN failures recorded but don't block
        failed_warn = [
            grader.grader_id
            for grader, outcome in warn_results
            if not outcome.passed
        ]
        if failed_warn:
            all_metrics["_failed_warn"] = len(failed_warn)
            feedbacks.insert(
                1 if failed_blocking else 0,  # After the single blocking-failure line
                f"WARNING: {', '.join(failed_warn)}",
            )

        overall_passed = all_blocking_passed

        return self.create_outcome(
            trial_id=transcript.task_id,
            passed=overall_passed,
            score=total_score,
            metrics=all_metrics,
            feedback="\n".join(feedbacks) if feedbacks else None,
        )
