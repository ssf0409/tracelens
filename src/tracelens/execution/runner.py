"""Evaluation runner for executing tasks with concurrency control.

The EvaluationRunner orchestrates:
- Running each task × run_index combination
- Concurrency limiting via semaphore
- Per-trial timeout enforcement
- Grading each trial's transcript
- Collecting results into a TrialBatch
"""

import asyncio
import logging
import traceback
from dataclasses import dataclass

from tracelens.core._time import utc_now
from tracelens.core.grader import Grader
from tracelens.core.outcome import Outcome
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import InfraError, Trial, TrialBatch, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter

logger = logging.getLogger(__name__)


# Exceptions that the runner treats as infrastructure failures (as opposed
# to task-level failures). Adapters can also raise ``InfraError`` explicitly
# for cases the runner can't infer from the exception type.
#
# The list is intentionally conservative: we only include errors that are
# almost always caused by the runtime (OOM, network, OS-level resource
# issues). Arbitrary exceptions stay classified as FAILED so a buggy agent
# doesn't silently inflate the infra-error rate and mask real regressions.
_INFRA_EXCEPTION_TYPES: tuple[type[BaseException], ...] = (
    InfraError,
    MemoryError,
    ConnectionError,
)


def _is_infra_exception(exc: BaseException) -> bool:
    """Return True if the exception should classify the trial as INFRA_ERROR."""
    return isinstance(exc, _INFRA_EXCEPTION_TYPES)


@dataclass
class RunnerConfig:
    """Configuration for the evaluation runner."""

    num_runs: int = 1
    max_concurrency: int = 5
    timeout_seconds: float = 300.0
    fail_fast: bool = False


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
    ) -> None:
        self.adapter = adapter
        self.graders = graders
        self.config = config or RunnerConfig()

    async def run(self, eval_set: EvalSet) -> TrialBatch:
        """Run all tasks × runs and grade results."""
        batch = TrialBatch(started_at=utc_now())
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        # Build work items: (task, run_index)
        work_items: list[tuple[Task, int]] = []
        for task in eval_set.tasks:
            for run_index in range(self.config.num_runs):
                work_items.append((task, run_index))

        # Run all concurrently with semaphore
        tasks = [
            self._run_one(task, run_index, semaphore, batch)
            for task, run_index in work_items
        ]
        await asyncio.gather(*tasks)

        batch.completed_at = utc_now()
        return batch

    async def _run_one(
        self,
        task: Task,
        run_index: int,
        semaphore: asyncio.Semaphore,
        batch: TrialBatch,
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
                is_infra = _is_infra_exception(exc)
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
                    is_infra = _is_infra_exception(exc)
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
