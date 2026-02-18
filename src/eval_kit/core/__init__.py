"""Core abstractions for the evaluation framework."""

from eval_kit.core.task import Task, TaskLoader, EvalSet, TaskExpectation
from eval_kit.core.trial import Trial, TrialStatus
from eval_kit.core.grader import (
    Grader,
    CodeGrader,
    LLMGrader,
    CompositeGrader,
    GraderType,
    GraderRole,
    GraderConfig,
)
from eval_kit.core.transcript import Transcript, TranscriptStep, StepType, ToolCall
from eval_kit.core.outcome import Outcome, GradeLevel
from eval_kit.core.decision_spec import (
    DecisionSpec,
    ModelConfig,
    PromptSpec,
    ToolSpec,
    AgentSpec,
    EnvironmentSpec,
)

__all__ = [
    "Task",
    "TaskLoader",
    "EvalSet",
    "TaskExpectation",
    "Trial",
    "TrialStatus",
    "Grader",
    "CodeGrader",
    "LLMGrader",
    "CompositeGrader",
    "GraderType",
    "GraderRole",
    "GraderConfig",
    "Transcript",
    "TranscriptStep",
    "StepType",
    "ToolCall",
    "Outcome",
    "GradeLevel",
    # Reproducibility
    "DecisionSpec",
    "ModelConfig",
    "PromptSpec",
    "ToolSpec",
    "AgentSpec",
    "EnvironmentSpec",
]
