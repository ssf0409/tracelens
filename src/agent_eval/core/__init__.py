"""Core abstractions for the evaluation framework."""

from agent_eval.core.task import Task, TaskLoader, EvalSet, TaskExpectation
from agent_eval.core.trial import Trial, TrialStatus
from agent_eval.core.grader import (
    Grader,
    CodeGrader,
    LLMGrader,
    CompositeGrader,
    GraderType,
    GraderRole,
    GraderConfig,
)
from agent_eval.core.transcript import Transcript, TranscriptStep, StepType, ToolCall
from agent_eval.core.outcome import Outcome, GradeLevel
from agent_eval.core.decision_spec import (
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
