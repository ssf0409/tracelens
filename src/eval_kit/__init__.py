"""eval-kit: Evaluation framework for AI agents.

A common evaluation framework for AI agents with support for:
- Code-based deterministic grading
- LLM-as-judge grading
- Human evaluation calibration
- Statistical analysis (pass@k, pass^k)
- Baseline regression detection
- CI/CD integration
"""

from eval_kit.core.decision_spec import (
    AgentSpec,
    DecisionSpec,
    EnvironmentSpec,
    ModelConfig,
    PromptSpec,
    ToolSpec,
)
from eval_kit.core.grader import (
    CodeGrader,
    CompositeGrader,
    Grader,
    GraderConfig,
    GraderRole,
    GraderType,
    LLMGrader,
)
from eval_kit.core.outcome import GradeLevel, Outcome
from eval_kit.core.task import EvalSet, Task, TaskLoader
from eval_kit.core.transcript import (
    StreamingEvent,
    StreamingEventType,
    Transcript,
    TranscriptStep,
)
from eval_kit.core.trial import Trial, TrialBatch, TrialStatus
from eval_kit.core.workflow import (
    StepResult,
    WorkflowContext,
    WorkflowStep,
    WorkflowTask,
)
from eval_kit.execution.agent_adapter import AgentAdapter, SimpleAdapter
from eval_kit.execution.runner import EvaluationRunner, RunnerConfig
from eval_kit.execution.workflow_runner import WorkflowAdapter, WorkflowRunner

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
    # Workflow
    "WorkflowTask",
    "WorkflowStep",
    "WorkflowContext",
    "StepResult",
    # Execution
    "AgentAdapter",
    "SimpleAdapter",
    "EvaluationRunner",
    "RunnerConfig",
    "WorkflowRunner",
    "WorkflowAdapter",
]
