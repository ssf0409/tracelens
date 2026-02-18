"""Trial models for execution tracking.

A Trial represents a single execution of a Task, including:
- The transcript of agent execution
- Grading outcomes from one or more graders
- Status and timing information
"""

from datetime import datetime
from enum import Enum
from typing import Any
import uuid

from pydantic import BaseModel, Field

from agent_eval.core.outcome import Outcome
from agent_eval.core.transcript import Transcript


class TrialStatus(str, Enum):
    """Status of a trial."""

    PENDING = "pending"      # Not yet started
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Finished successfully
    FAILED = "failed"        # Execution failed
    TIMEOUT = "timeout"      # Exceeded timeout
    SKIPPED = "skipped"      # Skipped (e.g., due to filter)


class Trial(BaseModel):
    """A single execution of a task.

    A Trial tracks:
    - Which task is being executed
    - The run index (for pass@k with multiple runs)
    - The execution transcript
    - Grading outcomes from all graders
    - Status and timing

    Example:
        trial = Trial(
            task_id=task.task_id,
            run_index=0,
            total_runs=5,
        )
        trial.status = TrialStatus.RUNNING
        trial.started_at = datetime.utcnow()
        # ... execute agent ...
        trial.transcript = transcript
        trial.status = TrialStatus.COMPLETED
        trial.completed_at = datetime.utcnow()
    """

    trial_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str

    # Run tracking (for pass@k)
    run_index: int = 0
    total_runs: int = 1

    # Status
    status: TrialStatus = TrialStatus.PENDING

    # Execution record
    transcript: Transcript | None = None

    # Grading results (can have multiple graders)
    outcomes: list[Outcome] = Field(default_factory=list)

    # Timing
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Error info
    error_message: str | None = None
    error_traceback: str | None = None

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """Calculate trial duration in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None

    @property
    def passed(self) -> bool:
        """Trial passes if ALL outcomes pass."""
        if not self.outcomes:
            return False
        return all(o.passed for o in self.outcomes)

    @property
    def aggregate_score(self) -> float | None:
        """Average score across all outcomes."""
        if not self.outcomes:
            return None
        return sum(o.score for o in self.outcomes) / len(self.outcomes)

    @property
    def is_complete(self) -> bool:
        """Check if trial has finished (successfully or not)."""
        return self.status in {
            TrialStatus.COMPLETED,
            TrialStatus.FAILED,
            TrialStatus.TIMEOUT,
            TrialStatus.SKIPPED,
        }

    @property
    def is_successful(self) -> bool:
        """Check if trial completed without errors."""
        return self.status == TrialStatus.COMPLETED and not self.error_message

    @property
    def fingerprint(self) -> str | None:
        """Get decision spec fingerprint from transcript."""
        if self.transcript and self.transcript.decision_spec:
            return self.transcript.decision_spec.fingerprint
        return None

    @property
    def fingerprint_short(self) -> str | None:
        """Get short decision spec fingerprint from transcript."""
        if self.transcript and self.transcript.decision_spec:
            return self.transcript.decision_spec.fingerprint_short
        return None

    def add_outcome(self, outcome: Outcome) -> None:
        """Add a grading outcome to this trial."""
        outcome.trial_id = self.trial_id
        self.outcomes.append(outcome)

    def get_outcome_by_grader(self, grader_id: str) -> Outcome | None:
        """Get outcome from a specific grader."""
        for outcome in self.outcomes:
            if outcome.grader_id == grader_id:
                return outcome
        return None

    def get_metric(self, metric_name: str) -> float | None:
        """Get a specific metric value from any outcome."""
        for outcome in self.outcomes:
            if metric_name in outcome.metrics:
                return outcome.metrics[metric_name]
        return None

    def to_summary_dict(self) -> dict[str, Any]:
        """Return a summary suitable for reporting."""
        summary = {
            "trial_id": self.trial_id,
            "task_id": self.task_id,
            "run_index": self.run_index,
            "status": self.status.value,
            "passed": self.passed,
            "score": self.aggregate_score,
            "duration_ms": self.duration_ms,
            "outcomes": [o.to_summary_dict() for o in self.outcomes],
        }
        if self.fingerprint_short:
            summary["fingerprint"] = self.fingerprint_short
        return summary

    def to_ci_dict(self) -> dict[str, Any]:
        """Return a compact dict for CI output."""
        result = {
            "task": self.task_id,
            "run": f"{self.run_index + 1}/{self.total_runs}",
            "status": self.status.value,
            "passed": self.passed,
            "score": round(self.aggregate_score, 4) if self.aggregate_score else None,
        }
        if self.fingerprint_short:
            result["fp"] = self.fingerprint_short
        return result


class TrialBatch(BaseModel):
    """Collection of trials for batch processing.

    Useful for running multiple trials in parallel and
    aggregating results.
    """

    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trials: list[Trial] = Field(default_factory=list)

    # Execution tracking
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def total_count(self) -> int:
        """Total number of trials."""
        return len(self.trials)

    @property
    def completed_count(self) -> int:
        """Number of completed trials."""
        return sum(1 for t in self.trials if t.is_complete)

    @property
    def passed_count(self) -> int:
        """Number of passed trials."""
        return sum(1 for t in self.trials if t.passed)

    @property
    def pass_rate(self) -> float:
        """Pass rate across all trials."""
        if not self.trials:
            return 0.0
        return self.passed_count / len(self.trials)

    @property
    def all_complete(self) -> bool:
        """Check if all trials are complete."""
        return all(t.is_complete for t in self.trials)

    def get_trials_for_task(self, task_id: str) -> list[Trial]:
        """Get all trials for a specific task."""
        return [t for t in self.trials if t.task_id == task_id]

    def get_pass_results_by_task(self) -> dict[str, list[bool]]:
        """Get pass/fail results grouped by task.

        Returns dict mapping task_id to list of boolean pass results.
        Useful for computing pass@k.
        """
        results: dict[str, list[bool]] = {}
        for trial in self.trials:
            if trial.task_id not in results:
                results[trial.task_id] = []
            results[trial.task_id].append(trial.passed)
        return results
