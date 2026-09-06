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
from typing import Any

from pydantic import ValidationError

from tracelens.core._time import utc_now
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.grader import Grader
from tracelens.core.outcome import Outcome
from tracelens.core.provenance import RunnerSettings, build_provenance
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


class _AdapterTimeoutError(Exception):
    """Wraps a TimeoutError raised inside adapter code.

    On Python >= 3.11 ``asyncio.TimeoutError`` (raised by the runner's
    budget via ``asyncio.wait_for``) and adapter-internal timeouts
    (``socket.timeout`` etc.) are the same ``TimeoutError`` type. Wrapping
    the adapter-raised one lets the runner keep TIMEOUT strictly for its
    own budget while adapter timeouts classify through
    ``infra_exception_types`` (FAILED by default).
    """

    def __init__(self, original: TimeoutError) -> None:
        super().__init__(str(original))
        self.original = original


class CheckpointError(Exception):
    """Raised when a checkpoint file is corrupt or belongs to a different run.

    Resuming merges checkpointed trials keyed on (task_id, run_index), so a
    checkpoint produced by a different eval set, adapter, or grader stack
    would silently mix results from two unrelated runs. Delete the checkpoint
    file (or point ``checkpoint_path`` at a new location) to start fresh.
    """


# Bump when the checkpoint file layout changes. Files without the envelope
# (written by TraceLens <= 0.3.x) are still readable but can't be
# identity-checked.
CHECKPOINT_FORMAT_VERSION = 1


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

    # Trials that end INFRA_ERROR are re-attempted up to this many extra
    # times. FAILED and TIMEOUT never retry: those are observations about
    # the agent, and retrying them would launder flakiness out of the pass
    # rate. The final trial records how many attempts it took in
    # ``Trial.attempts``.
    max_infra_retries: int = 0
    # Base delay before the first infra retry; doubles per attempt. The
    # concurrency slot is released while backing off.
    infra_retry_backoff_seconds: float = 1.0

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
        # Set per run(); written into checkpoints and compared on resume.
        self._checkpoint_identity: dict[str, Any] | None = None

    async def run(self, eval_set: EvalSet) -> TrialBatch:
        """Run all tasks × runs and grade results."""
        batch = TrialBatch(started_at=utc_now())
        # Record what is about to be measured (task content, graders, runner
        # settings) and which candidate is under test, before any trial
        # runs. The checkpoint identity is derived from the same record, so
        # there is one hashing rule (tracelens.core.provenance).
        provenance = build_provenance(
            eval_set=eval_set,
            adapter=self.adapter,
            graders=self.graders,
            settings=RunnerSettings.from_config(self.config),
            decision_spec=self.decision_spec,
            run_id=batch.batch_id,
            started_at=batch.started_at,
        )
        batch.provenance = provenance
        self._checkpoint_identity = {
            "eval_set_hash": provenance.measurement.eval_set_hash,
            "adapter": provenance.candidate.adapter.class_path,
            "graders": [g.class_path for g in provenance.measurement.graders],
            # The spec is the run's reproducibility identity; class paths
            # alone can't tell two SimpleAdapter/HTTPAPIAdapter configs
            # apart, so include the fingerprint whenever a spec is given.
            "decision_spec_fingerprint": provenance.candidate.decision_spec_fingerprint,
        }
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

        # Run all concurrently with semaphore. fail_fast shares one event:
        # once set, work items that haven't started yet produce no trials.
        stop_event = asyncio.Event() if self.config.fail_fast else None
        tasks = [
            self._run_one(task, run_index, semaphore, batch, total, stop_event)
            for task, run_index in work_items
        ]
        results = await asyncio.gather(*tasks)
        unrun = sum(1 for executed in results if not executed)
        if unrun:
            logger.warning(
                "fail_fast: %d work item(s) were not run after the first "
                "execution failure; a rerun with the same --checkpoint "
                "path executes them",
                unrun,
            )

        batch.completed_at = utc_now()
        provenance.completed_at = batch.completed_at
        self._save_checkpoint(batch)
        return batch

    def _load_resume_state(self, batch: TrialBatch) -> set[tuple[str, int]]:
        """Load completed trials from an existing checkpoint, if any.

        Returns the (task_id, run_index) keys to skip. Incomplete trials
        (e.g. RUNNING at crash time) and INFRA_ERROR trials are not loaded
        and will re-run — an infra error says nothing about the agent, so
        a resume is the natural moment to retry it.

        Raises:
            CheckpointError: if the file is unreadable as a checkpoint, or
                was written by a different eval set / adapter / graders.
        """
        completed: set[tuple[str, int]] = set()
        if not self.config.checkpoint_path:
            return completed

        path = Path(self.config.checkpoint_path)
        if not path.exists():
            return completed

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            raise CheckpointError(
                f"Corrupt or unreadable checkpoint file {path}: {exc}. "
                "Delete it, or point checkpoint_path somewhere else, to "
                "start fresh."
            ) from exc

        if isinstance(data, dict) and "batch" in data:
            version = data.get("version")
            if version != CHECKPOINT_FORMAT_VERSION:
                raise CheckpointError(
                    f"Checkpoint {path} has format version {version!r}; this "
                    f"TraceLens reads version {CHECKPOINT_FORMAT_VERSION}. "
                    "Delete the file or re-run with the matching version."
                )
            batch_data = data.get("batch")
            identity = data.get("identity")
            if not isinstance(identity, dict):
                raise CheckpointError(
                    f"Corrupt checkpoint file {path}: envelope is missing its "
                    "run identity."
                )
        else:
            # Bare-TrialBatch checkpoint written by TraceLens <= 0.3.x.
            logger.warning(
                "Checkpoint %s carries no run identity (pre-0.4 format); "
                "cannot verify it matches this eval set, adapter, and "
                "graders. Resuming anyway — delete the file if in doubt.",
                path,
            )
            batch_data = data
            identity = None

        try:
            loaded = TrialBatch.from_dict(batch_data)  # type: ignore[arg-type]
        except ValidationError as exc:
            raise CheckpointError(
                f"Corrupt checkpoint file {path}: does not contain a valid "
                f"trial batch ({exc})."
            ) from exc

        if identity is not None:
            self._validate_checkpoint_identity(identity, path)

        infra_reruns = 0
        for trial in loaded.trials:
            if not trial.is_complete:
                continue
            # INFRA_ERROR says nothing about the agent; SKIPPED is a
            # placeholder (e.g. from a fail-fast run) — both re-run.
            if trial.status in (TrialStatus.INFRA_ERROR, TrialStatus.SKIPPED):
                infra_reruns += 1
                continue
            batch.add_trial(trial)
            completed.add((trial.task_id, trial.run_index))
        logger.info(
            "Resumed %d completed trials from checkpoint %s",
            len(completed),
            path,
        )
        if infra_reruns:
            logger.info(
                "Re-running %d infra-errored trials from checkpoint %s",
                infra_reruns,
                path,
            )
        return completed

    def _validate_checkpoint_identity(self, identity: Any, path: Path) -> None:
        """Refuse to resume a checkpoint written by a different run setup."""
        current = self._checkpoint_identity
        assert current is not None  # set at the top of run()
        if not isinstance(identity, dict):
            raise CheckpointError(
                f"Corrupt checkpoint file {path}: malformed run identity."
            )
        mismatches: list[str] = []
        if identity.get("eval_set_hash") != current["eval_set_hash"]:
            mismatches.append(
                "eval set content (note: checkpointing requires stable, "
                "explicit task_ids — auto-generated ids change every run)"
            )
        if identity.get("decision_spec_fingerprint") != current[
            "decision_spec_fingerprint"
        ]:
            mismatches.append("decision spec")
        if identity.get("adapter") != current["adapter"]:
            mismatches.append(
                f"adapter ({identity.get('adapter')!r} vs {current['adapter']!r})"
            )
        # Order-insensitive: reordering graders doesn't change what was graded.
        if sorted(map(str, identity.get("graders") or [])) != sorted(
            current["graders"]
        ):
            mismatches.append("graders")
        if mismatches:
            raise CheckpointError(
                f"Checkpoint {path} was written by a different run — "
                f"mismatched {', '.join(mismatches)}. Refusing to resume: "
                "merging it would silently mix results from two runs. "
                "Delete the checkpoint file or use a different path."
            )

    async def _call_adapter_run(self, task: Task) -> Transcript:
        """Run the adapter, wrapping adapter-raised TimeoutError.

        Keeps the budget-timeout handler in ``_run_one`` from swallowing
        timeouts that came from inside the adapter (same exception type
        on Python >= 3.11).
        """
        try:
            return await self.adapter.run(task)
        except TimeoutError as exc:
            raise _AdapterTimeoutError(exc) from exc

    def _save_checkpoint(self, batch: TrialBatch) -> None:
        """Atomically persist the batch and its run identity."""
        if not self.config.checkpoint_path:
            return
        path = Path(self.config.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CHECKPOINT_FORMAT_VERSION,
            "identity": self._checkpoint_identity,
            "batch": batch.to_dict(),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)

    async def _run_one(
        self,
        task: Task,
        run_index: int,
        semaphore: asyncio.Semaphore,
        batch: TrialBatch,
        total: int,
        stop_event: asyncio.Event | None = None,
    ) -> bool:
        """Execute one (task, run_index) slot, retrying infra errors.

        Only INFRA_ERROR outcomes retry (up to ``max_infra_retries`` extra
        attempts, exponential backoff): they carry no signal about the
        agent. FAILED and TIMEOUT are agent observations and never retry.

        Returns True if the slot executed (a trial was recorded), False if
        fail_fast stopped it before it started — deliberately producing NO
        placeholder trial, so pass rates, the baseline gate, and
        checkpoints only see work that actually ran.
        """
        max_attempts = self.config.max_infra_retries + 1
        retried_errors: list[str] = []

        attempt = 1
        first = await self._attempt_trial(task, run_index, semaphore, stop_event)
        if first is None:
            return False
        trial = first
        while trial.status == TrialStatus.INFRA_ERROR and attempt < max_attempts:
            retried_errors.append(trial.error_message or "")
            backoff = self.config.infra_retry_backoff_seconds * 2 ** (attempt - 1)
            logger.warning(
                "Infra error on task %s run %d (attempt %d/%d): %s — "
                "retrying in %.1fs",
                task.task_id,
                run_index,
                attempt,
                max_attempts,
                trial.error_message,
                backoff,
            )
            if backoff > 0:
                await asyncio.sleep(backoff)
            retry = await self._attempt_trial(task, run_index, semaphore, stop_event)
            if retry is None:
                # Another trial tripped fail_fast mid-backoff; keep the
                # last real observation instead of discarding it.
                break
            trial = retry
            attempt += 1

        trial.attempts = attempt

        # fail_fast trips only on execution failures, only after retries
        # are exhausted, and never on teardown flakiness (the run itself
        # succeeded) or grading outcomes (graded below; a graded failure
        # is an agent-quality observation, not a broken harness).
        if (
            stop_event is not None
            and not stop_event.is_set()
            and trial.status
            in (TrialStatus.FAILED, TrialStatus.INFRA_ERROR, TrialStatus.TIMEOUT)
            and not trial.metadata.get("teardown_failed")
        ):
            stop_event.set()
            logger.warning(
                "fail_fast: task %s run %d ended %s — no new trials will start",
                task.task_id,
                run_index,
                trial.status.value,
            )
        if retried_errors:
            trial.metadata["infra_retry_errors"] = retried_errors

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

        return True

    async def _attempt_trial(
        self,
        task: Task,
        run_index: int,
        semaphore: asyncio.Semaphore,
        stop_event: asyncio.Event | None = None,
    ) -> Trial | None:
        """One execution attempt: lifecycle hooks, timeout, error handling.

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
            # Check under the semaphore: a work item that reaches its slot
            # after fail_fast tripped never starts (and records nothing).
            if stop_event is not None and stop_event.is_set():
                return None
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
                        self._call_adapter_run(task),
                        timeout=self.config.timeout_seconds,
                    )
                    if transcript.decision_spec is None and self.decision_spec is not None:
                        transcript.decision_spec = self.decision_spec
                    trial.transcript = transcript
                    trial.status = TrialStatus.COMPLETED
                except TimeoutError:
                    # Only the runner's own budget timeout lands here:
                    # adapter-raised TimeoutError is wrapped by
                    # _call_adapter_run so it classifies below instead.
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
                    if isinstance(exc, _AdapterTimeoutError):
                        exc = exc.original
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
                    # The run itself succeeded; record the distinction so
                    # fail_fast doesn't abort a suite over cleanup flakiness.
                    trial.metadata["teardown_failed"] = True
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
        return trial

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
