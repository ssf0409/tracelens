"""Issue #45: reported reliability must not depend on trial completion order.

The runner appends trials as they finish. Under concurrency, or after a
checkpoint resume, that order differs from ``run_index`` order, and pass^k
is a consecutive-window statistic, so the sequence handed to it must be
rebuilt in run order. These tests drive the real runner rather than
constructing batches by hand.
"""

import asyncio
from pathlib import Path

import pytest

from tracelens.core.grader import CodeGrader
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import InfraError, TrialBatch, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig
from tracelens.reporting.generator import ReportGenerator
from tracelens.statistics.consistency import pass_to_k


def _task(task_id: str) -> Task:
    return Task(task_id=task_id, name="t", description="d", input_data={"x": 1})


class _CallIndexGrader(CodeGrader):
    """Passes iff the adapter call index recorded in the transcript is in ``passing``."""

    def __init__(self, passing: set[int] | None = None) -> None:
        super().__init__("call_index")
        self._passing = passing or set()

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"call": float(transcript.final_output["call"])}

    def determine_pass(
        self, metrics: dict[str, float], task: Task
    ) -> tuple[bool, float]:
        ok = int(metrics["call"]) in self._passing
        return ok, 1.0 if ok else 0.0


class _GatedAdapter(AgentAdapter):
    """Each call blocks on its own event, so the test controls completion order."""

    def __init__(self, n_calls: int) -> None:
        self.calls = 0
        self.events = [asyncio.Event() for _ in range(n_calls)]

    async def run(self, task: Task) -> Transcript:
        call = self.calls
        self.calls += 1
        await self.events[call].wait()
        return Transcript(task_id=task.task_id, final_output={"call": call})


def test_concurrent_completion_order_does_not_change_pass_hat_k() -> None:
    """Runs finish in order 0, 2, 3, 1 but pass^2 follows run_index order."""
    release_order = iter([0, 2, 3, 1])
    adapter = _GatedAdapter(n_calls=4)

    def release_next(_done: int, _total: int) -> None:
        # Called after each trial is appended: let the next one finish.
        nxt = next(release_order, None)
        if nxt is not None:
            adapter.events[nxt].set()

    runner = EvaluationRunner(
        adapter=adapter,
        graders=[_CallIndexGrader(passing={0, 1})],  # calls 0 and 1 pass, 2 and 3 fail
        config=RunnerConfig(
            num_runs=4, max_concurrency=4, progress_callback=release_next
        ),
    )

    async def _run() -> TrialBatch:
        adapter.events[next(release_order)].set()
        return await runner.run(EvalSet(name="s", tasks=[_task("a")]))

    batch = asyncio.run(_run())

    completion_order = [t.run_index for t in batch.trials]
    by_run = sorted(batch.trials, key=lambda t: t.run_index)
    run_ordered = [t.passed for t in by_run]

    # Preconditions: the runner really did append out of run order, and the
    # call index the adapter saw is the run index (calls start in run order).
    assert completion_order == [0, 2, 3, 1]
    assert run_ordered == [True, True, False, False]

    # The property under test: every consumer sees run order.
    assert batch.get_pass_results_by_task() == {"a": run_ordered}
    report = ReportGenerator(k_values=[1], consistency_k_values=[2]).build_report(batch)
    assert report.reliability["pass^2"] == pytest.approx(1 / 3)  # TT, TF, FF
    # Completion order would have said something else.
    assert pass_to_k([t.passed for t in batch.trials], 2) == 0.0  # TF, FF, FT


class _InfraOnceAdapter(AgentAdapter):
    """Optionally raises InfraError on its first call, then records call indices.

    One class for both runs so the checkpoint identity (adapter class path)
    matches on resume.
    """

    def __init__(self, fail_first: bool) -> None:
        self.fail_first = fail_first
        self.calls = 0

    async def run(self, task: Task) -> Transcript:
        call = self.calls
        self.calls += 1
        if self.fail_first and call == 0:
            raise InfraError("first call hit infra")
        return Transcript(task_id=task.task_id, final_output={"call": call})


def test_checkpoint_resume_order_does_not_change_pass_hat_k(tmp_path: Path) -> None:
    """A re-run trial is appended last on resume; pass^k still follows run_index."""
    checkpoint = tmp_path / "checkpoint.json"
    eval_set = EvalSet(name="s", tasks=[_task("a")])

    def _runner(adapter: AgentAdapter) -> EvaluationRunner:
        return EvaluationRunner(
            adapter=adapter,
            graders=[_CallIndexGrader(passing={0, 2})],  # even call indices pass
            config=RunnerConfig(
                num_runs=3,
                max_concurrency=1,
                checkpoint_path=str(checkpoint),
                checkpoint_interval=1,
            ),
        )

    # First run, sequential: call 0 = run 0 hits infra; run 1 fails; run 2 passes.
    first = asyncio.run(_runner(_InfraOnceAdapter(fail_first=True)).run(eval_set))
    statuses = {t.run_index: t.status for t in first.trials}
    assert statuses == {
        0: TrialStatus.INFRA_ERROR,
        1: TrialStatus.COMPLETED,
        2: TrialStatus.COMPLETED,
    }

    # Resume: only run 0 is re-run (fresh adapter, call 0 -> passes) and it is
    # appended after the trials loaded from the checkpoint.
    resumed = asyncio.run(_runner(_InfraOnceAdapter(fail_first=False)).run(eval_set))
    assert [t.run_index for t in resumed.trials] == [1, 2, 0]

    run_ordered = [True, False, True]
    assert resumed.get_pass_results_by_task() == {"a": run_ordered}
    report = ReportGenerator(k_values=[1], consistency_k_values=[2]).build_report(resumed)
    assert report.reliability["pass^2"] == 0.0  # TF, FT
    # Completion order F, T, T would have reported 1/2.
    assert pass_to_k([t.passed for t in resumed.trials], 2) == 0.5
