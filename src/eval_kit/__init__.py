"""eval-kit: Evaluation framework for AI agents.

A common evaluation framework for AI agents with support for:
- Code-based deterministic grading
- LLM-as-judge grading
- Human evaluation calibration
- Statistical analysis (pass@k, pass^k)
- Baseline regression detection
- CI/CD integration
"""

from eval_kit.core.task import Task, TaskLoader, EvalSet
from eval_kit.core.trial import Trial, TrialBatch, TrialStatus
from eval_kit.core.grader import (
    Grader,
    CodeGrader,
    LLMGrader,
    CompositeGrader,
    GraderType,
    GraderRole,
    GraderConfig,
)
from eval_kit.core.transcript import Transcript, TranscriptStep
from eval_kit.core.outcome import Outcome, GradeLevel
from eval_kit.core.decision_spec import (
    DecisionSpec,
    ModelConfig,
    PromptSpec,
    ToolSpec,
    AgentSpec,
    EnvironmentSpec,
)
from eval_kit.execution.agent_adapter import AgentAdapter, SimpleAdapter
from eval_kit.execution.runner import EvaluationRunner, RunnerConfig

__version__ = "0.1.0"

__all__ = [
    # Core models
    "Task",
    "TaskLoader",
    "EvalSet",
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
    "Outcome",
    "GradeLevel",
    # Reproducibility
    "DecisionSpec",
    "ModelConfig",
    "PromptSpec",
    "ToolSpec",
    "AgentSpec",
    "EnvironmentSpec",
    # Execution
    "AgentAdapter",
    "SimpleAdapter",
    "EvaluationRunner",
    "RunnerConfig",
]
