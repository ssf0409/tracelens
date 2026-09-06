"""Trial models for execution tracking.

A Trial represents a single execution of a Task, including:
- The transcript of agent execution
- Grading outcomes from one or more graders
- Status and timing information
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from tracelens.core.outcome import Outcome
from tracelens.core.provenance import RunProvenance
from tracelens.core.transcript import Transcript


class TrialStatus(str, Enum):
    """Status of a trial."""

    PENDING = "pending"            # Not yet started
    RUNNING = "running"            # Currently executing
    COMPLETED = "completed"        # Finished successfully
    FAILED = "failed"              # Task-level failure (agent couldn't solve it)
    INFRA_ERROR = "infra_error"    # Infrastructure failure (OOM, network, sandbox)
    TIMEOUT = "timeout"             # Runner budget timeout (adapter-raised timeouts classify as FAILED/INFRA_ERROR)
    SKIPPED = "skipped"             # Skipped (e.g., due to filter)


# Trial states that are evidence about the agent (statistical contract).
_GRADABLE_STATUSES = frozenset({
    TrialStatus.COMPLETED,
    TrialStatus.FAILED,
    TrialStatus.TIMEOUT,
})


class InfraError(Exception):
    """Raised by adapters when a failure is known to be infrastructural.

    Separating infra failures from task-level failures matters because they
    mean different things for evaluation scores. A pod killed for exceeding
    its memory limit tells you nothing about the agent's capability — but it
    *does* tell you the eval's resource configuration is too tight (see
    Anthropic's "Quantifying infrastructure noise in agentic coding evals",
    which measured infra error rates dropping from 5.8% at strict enforcement
    to 0.5% uncapped).

    When the runner catches this exception — or ``MemoryError`` /
    ``ConnectionError``, the conservative default set — the trial's status
    is set to ``TrialStatus.INFRA_ERROR`` rather than ``FAILED``, and the
    infra error rate is surfaced separately in reports so you can decide
    whether a regression is real or a noise artefact. Broader classes
    (``OSError``, ``TimeoutError``, custom rate-limit errors, ...) are
    downstream policy: add them via ``RunnerConfig.infra_exception_types``
    or the CLI's ``--infra-exceptions``. Only the runner's own budget
    timeout is classified ``TrialStatus.TIMEOUT``; a ``TimeoutError``
    raised inside the adapter classifies through the configurable set
    (``FAILED`` by default).

    Example:
        class MyAdapter(AgentAdapter):
            async def run(self, task):
                try:
                    return await do_work(task)
                except httpx.ConnectError as exc:
                    raise InfraError(f"upstream API unreachable: {exc}") from exc
    """


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
        trial.started_at = utc_now()
        # ... execute agent ...
        trial.transcript = transcript
        trial.status = TrialStatus.COMPLETED
        trial.completed_at = utc_now()
    """

    trial_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str

    # Run tracking (for pass@k)
    run_index: int = 0
    total_runs: int = 1

    # Number of execution attempts behind this trial. >1 means earlier
    # attempts hit infra errors and were retried (RunnerConfig.max_infra_retries);
    # the retried-away error messages are kept in metadata["infra_retry_errors"].
    attempts: int = 1

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
            TrialStatus.INFRA_ERROR,
            TrialStatus.TIMEOUT,
            TrialStatus.SKIPPED,
        }

    @property
    def is_successful(self) -> bool:
        """Check if trial completed without errors."""
        return self.status == TrialStatus.COMPLETED and not self.error_message

    @property
    def has_grader_error(self) -> bool:
        """Whether any grader crashed while grading this trial.

        Grader crashes are synthesized as failed outcomes so the trial
        stays conservative (not passed), but they must be counted
        separately — they measure the eval harness, not the agent.
        """
        return any(o.grader_error for o in self.outcomes)

    @property
    def is_infra_failure(self) -> bool:
        """Whether this trial failed due to infrastructure, not the agent.

        Infra failures (OOM kills, network errors, sandbox terminations)
        should be counted separately from task failures when interpreting
        scores — otherwise noisy infra inflates the apparent failure rate.
        """
        return self.status == TrialStatus.INFRA_ERROR

    @property
    def is_gradable(self) -> bool:
        """Whether this trial counts as agent evidence.

        COMPLETED, FAILED, and TIMEOUT trials are evidence about the agent
        (the latter two as failures). INFRA_ERROR trials, trials where a
        grader crashed, and trials that never ran (PENDING, RUNNING,
        SKIPPED) are not: they are excluded from pass rates, pass@k, and
        pass^k and reported separately. See ``docs/statistical-contract.md``.
        """
        return self.status in _GRADABLE_STATUSES and not self.has_grader_error

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

    # What was measured (task content, graders, runner settings) and which
    # candidate was under test; recorded by ``EvaluationRunner``. ``None``
    # for batches built by hand or loaded from artifacts written before
    # provenance existed: nothing is invented for them.
    provenance: RunProvenance | None = None

    @property
    def total_count(self) -> int:
        """Total number of trials."""
        return len(self.trials)

    @property
    def completed_count(self) -> int:
        """Number of completed trials."""
        return sum(1 for t in self.trials if t.is_complete)

    @property
    def gradable_count(self) -> int:
        """Number of trials that are agent evidence (``Trial.is_gradable``)."""
        return sum(1 for t in self.trials if t.is_gradable)

    @property
    def excluded_count(self) -> int:
        """Trials excluded from agent statistics: harness failures and never-run trials."""
        return self.total_count - self.gradable_count

    @property
    def passed_count(self) -> int:
        """Number of passed gradable trials."""
        return sum(1 for t in self.trials if t.is_gradable and t.passed)

    @property
    def pass_rate(self) -> float:
        """Passed gradable trials divided by gradable trials.

        Harness failures (INFRA_ERROR, grader crashes) and trials that never
        ran are not in the denominator; they are surfaced separately via
        ``infra_error_rate``, ``grader_error_rate``, and ``excluded_count``.
        Returns 0.0 when there is no gradable trial; report renderers show
        that case as N/A using ``gradable_count``.
        """
        gradable = self.gradable_count
        if not gradable:
            return 0.0
        return self.passed_count / gradable

    @property
    def infra_error_count(self) -> int:
        """Number of trials that failed due to infrastructure issues."""
        return sum(1 for t in self.trials if t.is_infra_failure)

    @property
    def infra_error_rate(self) -> float:
        """Fraction of trials that hit infrastructure failures.

        Report this alongside ``pass_rate``: Anthropic's infra-noise study
        found that infra error rates can move by 5+ percentage points
        purely from resource-configuration changes. A spike in
        ``infra_error_rate`` between two baselines is a strong hint that
        the regression is noise, not a real capability drop.
        """
        if not self.trials:
            return 0.0
        return self.infra_error_count / len(self.trials)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens across all trial transcripts."""
        return sum(
            t.transcript.input_tokens for t in self.trials if t.transcript is not None
        )

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens across all trial transcripts."""
        return sum(
            t.transcript.output_tokens for t in self.trials if t.transcript is not None
        )

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all trial transcripts."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def grader_error_count(self) -> int:
        """Number of trials where at least one grader crashed."""
        return sum(1 for t in self.trials if t.has_grader_error)

    @property
    def grader_error_rate(self) -> float:
        """Fraction of trials affected by grader crashes.

        Report this alongside ``pass_rate``: a spike here means the
        grading harness is broken, not that the agent regressed.
        """
        if not self.trials:
            return 0.0
        return self.grader_error_count / len(self.trials)

    @property
    def all_complete(self) -> bool:
        """Check if all trials are complete."""
        return all(t.is_complete for t in self.trials)

    def add_trial(self, trial: Trial) -> None:
        """Add a trial to the batch."""
        self.trials.append(trial)

    def get_trials_for_task(self, task_id: str) -> list[Trial]:
        """Get all trials for a specific task."""
        return [t for t in self.trials if t.task_id == task_id]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict with full round-trip fidelity."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialBatch":
        """Reconstruct a TrialBatch from a dict produced by to_dict()."""
        return cls.model_validate(data)

    def _trials_by_task_in_run_order(self) -> dict[str, list[Trial]]:
        """Group trials by task, ordered by ``run_index``.

        The runner appends trials in completion order, which under
        concurrency or after a checkpoint resume differs from run order.
        Every per-task sequence handed to the statistics layer is rebuilt
        from ``run_index`` so that reported numbers never depend on timing.

        Raises:
            ValueError: If two trials share the same ``(task_id, run_index)``
                or a ``run_index`` is negative. Both make the run sequence
                ambiguous, so they are rejected rather than silently merged.
        """
        grouped: dict[str, dict[int, Trial]] = {}
        for trial in self.trials:
            if trial.run_index < 0:
                raise ValueError(
                    f"Negative run_index {trial.run_index} for task "
                    f"{trial.task_id!r} (trial {trial.trial_id})"
                )
            per_task = grouped.setdefault(trial.task_id, {})
            if trial.run_index in per_task:
                raise ValueError(
                    f"Duplicate run_index {trial.run_index} for task "
                    f"{trial.task_id!r}: trials "
                    f"{per_task[trial.run_index].trial_id} and {trial.trial_id}. "
                    "Give each trial of a task a distinct run_index "
                    "(EvaluationRunner does this automatically)."
                )
            per_task[trial.run_index] = trial
        return {
            task_id: [per_task[i] for i in sorted(per_task)]
            for task_id, per_task in grouped.items()
        }

    def get_pass_results_by_task(self) -> dict[str, list[bool]]:
        """Get pass/fail results grouped by task, in ``run_index`` order.

        Returns dict mapping task_id to a list of boolean pass results
        ordered by ``run_index`` regardless of the order trials were
        appended. Only gradable trials are included (``Trial.is_gradable``);
        a task whose trials were all excluded maps to an empty list. Missing
        or excluded run indices are simply absent from the list; use
        :meth:`get_pass_sequences_by_task` when gaps matter (pass^k).
        Useful for computing pass@k.

        Raises:
            ValueError: If two trials share the same ``(task_id, run_index)``.
        """
        return {
            task_id: [t.passed for t in trials if t.is_gradable]
            for task_id, trials in self._trials_by_task_in_run_order().items()
        }

    def get_pass_sequences_by_task(self) -> dict[str, list[bool | None]]:
        """Pass/fail sequences indexed by ``run_index``, with ``None`` for gaps.

        Position ``i`` of each list is the outcome of ``run_index == i``; a
        run index with no trial, or whose trial is not gradable (a harness
        failure), yields ``None``. Consecutive-window statistics (pass^k)
        consume this shape so that runs on either side of a gap are never
        treated as consecutive observations.

        Raises:
            ValueError: If two trials share the same ``(task_id, run_index)``.
        """
        sequences: dict[str, list[bool | None]] = {}
        for task_id, trials in self._trials_by_task_in_run_order().items():
            seq: list[bool | None] = [None] * (trials[-1].run_index + 1)
            for trial in trials:
                seq[trial.run_index] = trial.passed if trial.is_gradable else None
            sequences[task_id] = seq
        return sequences
