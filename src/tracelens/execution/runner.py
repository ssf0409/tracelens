"""Evaluation runner for executing tasks with concurrency control.

The EvaluationRunner orchestrates:
- Running each task × run_index combination
- Concurrency limiting via semaphore
- Per-trial timeout enforcement
- Grading each trial's transcript
- Collecting results into a TrialBatch
"""

import asyncio
import json
import logging
import os
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from tracelens.core._time import utc_now
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.grader import Grader
from tracelens.core.outcome import Outcome
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import InfraError, Trial, TrialBatch, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter

logger = logging.getLogger(__name__)


# Default exceptions that the runner treats as infrastructure failures (as
# opposed to task-level failures). Adapters can also raise ``InfraError``
# explicitly for cases the runner can't infer from the exception type.
#
# The default is intentionally conservative: only errors that are almost
# always caused by the runtime (OOM, network). Broad classes like OSError
# are excluded because their subclasses (FileNotFoundError, PermissionError)
# are usually agent bugs — counting those as infra would silently inflate
# the infra-error rate and mask real regressions. Which types count as
# infra is downstream policy: extend the set per project via
# ``RunnerConfig.infra_exception_types`` (or ``--infra-exceptions`` on the
# CLI), e.g. ``DEFAULT_INFRA_EXCEPTION_TYPES + (OSError,)`` for evals on
# shared runners where disk-full is an environment problem.
DEFAULT_INFRA_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    InfraError,
    MemoryError,
    ConnectionError,
)


@dataclass
class RunnerConfig:
    """Configuration for the evaluation runner."""

    num_runs: int = 1
    max_concurrency: int = 5
    timeout_seconds: float = 300.0
    fail_fast: bool = False

    # Exception types classified as INFRA_ERROR instead of FAILED. The
    # conservative default is DEFAULT_INFRA_EXCEPTION_TYPES; extend it when
    # your environment makes broader classes unambiguous infra, e.g.
    # ``DEFAULT_INFRA_EXCEPTION_TYPES + (OSError,)``. The runner's own
    # budget timeout is classified TIMEOUT before this set is consulted.
    infra_exception_types: tuple[type[BaseException], ...] = (
        DEFAULT_INFRA_EXCEPTION_TYPES
    )

    # Called as (completed_trials, total_trials) after each trial finishes.
    progress_callback: Callable[[int, int], None] | None = None

    # When set, the batch is persisted here every ``checkpoint_interval``
    # trials (and once at the end). A later run with the same path resumes:
    # trials already complete in the checkpoint are loaded instead of re-run.
    checkpoint_path: str | None = None
    checkpoint_interval: int = 10


class EvaluationRunner:
    """Runs evaluations with concurrency control and timeout enforcement.

    Example:
        runner = EvaluationRunner(
            adapter=my_adapter,
            graders=[quality_grader, safety_grader],
            config=RunnerConfig(num_runs=5, max_concurrency=10),
        )
        batch = await runner.run(eval_set)
        print(f"Pass rate: {batch.pass_rate:.2%}")
    """

    def __init__(
        self,
        adapter: AgentAdapter,
        graders: list[Grader],
        config: RunnerConfig | None = None,
        decision_spec: DecisionSpec | None = None,
    ) -> None:
        self.adapter = adapter
        self.graders = graders
        self.config = config or RunnerConfig()
        # Stamped onto every transcript that doesn't already carry one,
        # so baselines can be compared by reproducibility fingerprint.
        self.decision_spec = decision_spec

    async def run(self, eval_set: EvalSet) -> TrialBatch:
        """Run all tasks × runs and grade results."""
        batch = TrialBatch(started_at=utc_now())
        completed_keys = self._load_resume_state(batch)
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        # Build work items: (task, run_index), skipping trials already
        # completed in a resumed checkpoint.
        work_items: list[tuple[Task, int]] = []
        for task in eval_set.tasks:
            for run_index in range(self.config.num_runs):
                if (task.task_id, run_index) not in completed_keys:
                    work_items.append((task, run_index))

        total = len(work_items) + len(completed_keys)

        # Run all concurrently with semaphore
        tasks = [
            self._run_one(task, run_index, semaphore, batch, total)
            for task, run_index in work_items
        ]
        await asyncio.gather(*tasks)

        batch.completed_at = utc_now()
        self._save_checkpoint(batch)
        return batch

    def _load_resume_state(self, batch: TrialBatch) -> set[tuple[str, int]]:
        """Load completed trials from an existing checkpoint, if any.

        Returns the (task_id, run_index) keys to skip. Incomplete trials
        (e.g. RUNNING at crash time) are not loaded and will re-run.
        """
        completed: set[tuple[str, int]] = set()
        if not self.config.checkpoint_path:
            return completed

        path = Path(self.config.checkpoint_path)
        if not path.exists():
            return completed

        loaded = TrialBatch.from_dict(json.loads(path.read_text()))
        for trial in loaded.trials:
            if trial.is_complete:
                batch.add_trial(trial)
                completed.add((trial.task_id, trial.run_index))
        logger.info(
            "Resumed %d completed trials from checkpoint %s",
            len(completed),
            path,
        )
        return completed

    def _save_checkpoint(self, batch: TrialBatch) -> None:
        """Atomically persist the batch to the checkpoint path."""
        if not self.config.checkpoint_path:
            return
        path = Path(self.config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(batch.to_dict()))
        os.replace(tmp, path)

    async def _run_one(
        self,
        task: Task,
        run_index: int,
        semaphore: asyncio.Semaphore,
        batch: TrialBatch,
        total: int,
    ) -> None:
        """Execute a single trial with lifecycle hooks, timeout, and error handling.

        Lifecycle: setup → run → teardown (always called).
        If setup fails, run is skipped but teardown still runs.
        If teardown fails on an otherwise-successful trial, the trial is marked FAILED.
        """
        trial = Trial(
            task_id=task.task_id,
            run_index=run_index,
            total_runs=self.config.num_runs,
            status=TrialStatus.RUNNING,
            started_at=utc_now(),
        )

        transcript: Transcript | None = None

        async with semaphore:
            setup_failed = False

            # --- setup ---
            try:
                await self.adapter.setup(task)
            except Exception as exc:
                setup_failed = True
                is_infra = isinstance(exc, self.config.infra_exception_types)
                trial.status = (
                    TrialStatus.INFRA_ERROR if is_infra else TrialStatus.FAILED
                )
                trial.error_message = f"Setup failed: {exc}"
                trial.error_traceback = traceback.format_exc()
                logger.error(
                    "Setup %s for task %s run %d: %s",
                    "hit an infra error" if is_infra else "failed",
                    task.task_id,
                    run_index,
                    exc,
                )

            # --- run (skipped if setup failed) ---
            if not setup_failed:
                try:
                    transcript = await asyncio.wait_for(
                        self.adapter.run(task),
                        timeout=self.config.timeout_seconds,
                    )
                    if transcript.decision_spec is None and self.decision_spec is not None:
                        transcript.decision_spec = self.decision_spec
                    trial.transcript = transcript
                    trial.status = TrialStatus.COMPLETED
                except TimeoutError:
                    trial.status = TrialStatus.TIMEOUT
                    trial.error_message = (
                        f"Trial timed out after {self.config.timeout_seconds}s"
                    )
                    logger.warning(
                        "Trial timed out for task %s run %d after %.1fs",
                        task.task_id,
                        run_index,
                        self.config.timeout_seconds,
                    )
                except Exception as exc:
                    is_infra = isinstance(exc, self.config.infra_exception_types)
                    trial.status = (
                        TrialStatus.INFRA_ERROR if is_infra else TrialStatus.FAILED
                    )
                    trial.error_message = str(exc)
                    trial.error_traceback = traceback.format_exc()
                    logger.error(
                        "Agent execution %s for task %s run %d: %s",
                        "hit an infra error" if is_infra else "failed",
                        task.task_id,
                        run_index,
                        exc,
                    )

            # --- teardown (always called) ---
            try:
                await self.adapter.teardown(task, transcript)
            except Exception as teardown_exc:
                if trial.status == TrialStatus.COMPLETED:
                    trial.status = TrialStatus.FAILED
                    trial.error_message = (
                        f"Teardown failed: {teardown_exc}"
                    )
                    trial.error_traceback = traceback.format_exc()
                else:
                    trial.error_message = (
                        f"{trial.error_message}; "
                        f"Teardown also failed: {teardown_exc}"
                    )
                logger.error(
                    "Teardown failed for task %s run %d: %s",
                    task.task_id,
                    run_index,
                    teardown_exc,
                )

        trial.completed_at = utc_now()

        # Grade if we have a transcript
        if trial.transcript is not None:
            await self._grade_trial(trial, task)

        batch.add_trial(trial)

        if self.config.progress_callback is not None:
            self.config.progress_callback(len(batch.trials), total)

        if (
            self.config.checkpoint_path
            and self.config.checkpoint_interval > 0
            and len(batch.trials) % self.config.checkpoint_interval == 0
        ):
            self._save_checkpoint(batch)

    async def _grade_trial(self, trial: Trial, task: Task) -> None:
        """Run all graders on a trial's transcript."""
        assert trial.transcript is not None
        for grader in self.graders:
            try:
                outcome = await grader.grade(trial.transcript, task)
                trial.add_outcome(outcome)
            except MemoryError:
                # Known-corrupt process state: propagate instead of
                # recording bogus 0-score outcomes for the rest of the run.
                raise
            except Exception as exc:
                # Grader failure → failed outcome (not an agent failure)
                logger.error(
                    "Grader %s crashed on trial %s: %s",
                    grader.grader_id,
                    trial.trial_id,
                    exc,
                )
                trial.add_outcome(Outcome(
                    trial_id=trial.trial_id,
                    grader_id=grader.grader_id,
                    passed=False,
                    score=0.0,
                    metrics={"_grader_error": 1.0},
                    feedback=f"GRADER CRASH (not an agent failure): {exc}",
                    grader_error=True,
                ))
