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
from agent_eval.core.grader import Grader, CodeGrader, LLMGrader, GraderType
from agent_eval.core.transcript import Transcript, TranscriptStep
from agent_eval.core.outcome import Outcome, GradeLevel

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
    "GraderType",
    "Transcript",
    "TranscriptStep",
    "Outcome",
    "GradeLevel",
]
