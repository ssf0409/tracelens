"""Agent Evaluation Framework.

A common evaluation framework for AI agents with support for:
- Code-based deterministic grading
- LLM-as-judge grading
- Human evaluation calibration
- Statistical analysis (pass@k, pass^k)
- Baseline regression detection
- CI/CD integration
"""

from agent_eval.core.task import Task, TaskLoader, EvalSet
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
from agent_eval.core.transcript import Transcript, TranscriptStep
from agent_eval.core.outcome import Outcome, GradeLevel
from agent_eval.core.decision_spec import (
    DecisionSpec,
    ModelConfig,
    PromptSpec,
    ToolSpec,
    AgentSpec,
    EnvironmentSpec,
)

__version__ = "0.1.0"

__all__ = [
    # Core models
    "Task",
    "TaskLoader",
    "EvalSet",
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
