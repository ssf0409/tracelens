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
from datetime import datetime

from eval_kit.core.grader import Grader
from eval_kit.core.outcome import Outcome
from eval_kit.core.task import EvalSet, Task
from eval_kit.core.transcript import Transcript
from eval_kit.core.trial import Trial, TrialBatch, TrialStatus
from eval_kit.execution.agent_adapter import AgentAdapter

logger = logging.getLogger(__name__)


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
        batch = TrialBatch(started_at=datetime.utcnow())
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

        batch.completed_at = datetime.utcnow()
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
            started_at=datetime.utcnow(),
        )

        transcript: Transcript | None = None

        async with semaphore:
            setup_failed = False

            # --- setup ---
            try:
                await self.adapter.setup(task)
            except Exception as exc:
                setup_failed = True
                trial.status = TrialStatus.FAILED
                trial.error_message = f"Setup failed: {exc}"
                trial.error_traceback = traceback.format_exc()
                logger.error(
                    "Setup failed for task %s run %d: %s",
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
                except Exception as exc:
                    trial.status = TrialStatus.FAILED
                    trial.error_message = str(exc)
                    trial.error_traceback = traceback.format_exc()
                    logger.error(
                        "Agent execution failed for task %s run %d: %s",
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

        trial.completed_at = datetime.utcnow()

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
                ))
