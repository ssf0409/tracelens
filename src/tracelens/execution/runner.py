"""Evaluation runner for executing tasks with concurrency control.

The EvaluationRunner orchestrates:
- Running each task × run_index combination
- Concurrency limiting via semaphore
- Per-trial timeout enforcement
- Grading each trial's transcript
- Collecting results into a TrialBatch
"""

import asyncio
import hashlib
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


def _class_path(obj: object) -> str:
    """Dotted import path of an object's class, for checkpoint identity."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _eval_set_fingerprint(eval_set: EvalSet) -> str:
    """Content hash of the eval set's tasks.

    Covers task ids, inputs, expectations, and metadata — the things that
    make checkpointed (task_id, run_index) results meaningful. Deliberately
    excludes EvalSet-level metadata (timestamps, name) and num_runs, so
    resuming with more runs of the same tasks still works.
    """
    tasks = sorted(
        (t.model_dump(mode="json") for t in eval_set.tasks),
        key=lambda d: str(d.get("task_id")),
    )
    canonical = json.dumps(tasks, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


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
        self._checkpoint_identity = {
            "eval_set_hash": _eval_set_fingerprint(eval_set),
            "adapter": _class_path(self.adapter),
            "graders": [_class_path(g) for g in self.graders],
            # The spec is the run's reproducibility identity; class paths
            # alone can't tell two SimpleAdapter/HTTPAPIAdapter configs
            # apart, so include the fingerprint whenever a spec is given.
            "decision_spec_fingerprint": (
                self.decision_spec.fingerprint if self.decision_spec else None
            ),
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
    ) -> None:
        """Execute one (task, run_index) slot, retrying infra errors.

        Only INFRA_ERROR outcomes retry (up to ``max_infra_retries`` extra
        attempts, exponential backoff): they carry no signal about the
        agent. FAILED and TIMEOUT are agent observations and never retry.
        """
        max_attempts = self.config.max_infra_retries + 1
        retried_errors: list[str] = []

        attempt = 1
        trial = await self._attempt_trial(task, run_index, semaphore)
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
            trial = await self._attempt_trial(task, run_index, semaphore)
            attempt += 1

        trial.attempts = attempt
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

    async def _attempt_trial(
        self,
        task: Task,
        run_index: int,
        semaphore: asyncio.Semaphore,
    ) -> Trial:
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
