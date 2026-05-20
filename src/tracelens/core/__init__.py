"""Core abstractions for the evaluation framework."""

from tracelens.core.decision_spec import (
    AgentSpec,
    DecisionSpec,
    EnvironmentSpec,
    ModelConfig,
    PromptSpec,
    ToolSpec,
)
from tracelens.core.grader import (
    CodeGrader,
    CompositeGrader,
    Grader,
    GraderConfig,
    GraderRole,
    GraderType,
    LLMGrader,
)
from tracelens.core.outcome import GradeLevel, Outcome
from tracelens.core.task import EvalSet, Task, TaskExpectation, TaskLoader
from tracelens.core.transcript import (
    StepType,
    StreamingEvent,
    StreamingEventType,
    ToolCall,
    Transcript,
    TranscriptStep,
)
from tracelens.core.trial import Trial, TrialBatch, TrialStatus

__all__ = [
    "Task",
    "TaskLoader",
    "EvalSet",
    "TaskExpectation",
    "Trial",
    "TrialBatch",
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
    "StreamingEvent",
    "StreamingEventType",
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
