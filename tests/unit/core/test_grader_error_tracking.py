"""Tests for first-class grader error tracking.

A grader crash is an infrastructure problem, not an agent failure.
These tests pin down that crashes are marked on the Outcome
(``grader_error=True``), surfaced in batch statistics and reports,
and that known-corrupt states (MemoryError) propagate instead of
being silently converted into agent failures.
"""

import asyncio

import pytest

from tracelens.core.grader import CodeGrader, CompositeGrader
from tracelens.core.outcome import Outcome
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch
from tracelens.execution.agent_adapter import SimpleAdapter
from tracelens.execution.runner import EvaluationRunner
from tracelens.reporting.generator import ReportGenerator


class _PassGrader(CodeGrader):
    def __init__(self) -> None:
        super().__init__("pass_grader")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"quality": 0.9}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return True, 0.9


class _CrashingGrader(CodeGrader):
    def __init__(self, exc: Exception | None = None) -> None:
        super().__init__("crashing_grader")
        self._exc = exc or RuntimeError("boom")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        raise self._exc

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return False, 0.0


def _task(task_id: str = "t1") -> Task:
    return Task(task_id=task_id, name="t", description="d", input_data={"x": 1})


def _transcript(task_id: str = "t1") -> Transcript:
    return Transcript(task_id=task_id)


def _outcome(*, grader_error: bool = False, passed: bool = False) -> Outcome:
    return Outcome(
        trial_id="tr1",
        grader_id="g",
        passed=passed,
        score=0.0,
        grader_error=grader_error,
    )


def test_outcome_grader_error_defaults_false() -> None:
    outcome = Outcome(trial_id="tr1", grader_id="g", passed=True, score=1.0)
    assert outcome.grader_error is False


def test_composite_marks_grader_error_on_crash() -> None:
    composite = CompositeGrader(
        "composite",
        graders=[(_PassGrader(), 1.0), (_CrashingGrader(), 1.0)],
    )

    outcome = asyncio.run(composite.grade(_transcript(), _task()))

    assert outcome.grader_error is True
    assert outcome.passed is False or outcome.passed is True  # passing not asserted here
    assert "crashed" in (outcome.feedback or "")


def test_composite_no_grader_error_when_all_succeed() -> None:
    composite = CompositeGrader("composite", graders=[(_PassGrader(), 1.0)])

    outcome = asyncio.run(composite.grade(_transcript(), _task()))

    assert outcome.grader_error is False


def test_composite_reraises_memory_error() -> None:
    composite = CompositeGrader(
        "composite",
        graders=[(_CrashingGrader(MemoryError("oom")), 1.0)],
    )

    with pytest.raises(MemoryError):
        asyncio.run(composite.grade(_transcript(), _task()))


def test_trial_has_grader_error_property() -> None:
    trial = Trial(task_id="t1")
    trial.add_outcome(_outcome(grader_error=False, passed=True))
    assert trial.has_grader_error is False

    trial.add_outcome(_outcome(grader_error=True))
    assert trial.has_grader_error is True


def test_batch_grader_error_count_and_rate() -> None:
    batch = TrialBatch()

    clean = Trial(task_id="t1")
    clean.add_outcome(_outcome(passed=True))
    errored = Trial(task_id="t2")
    errored.add_outcome(_outcome(grader_error=True))

    batch.add_trial(clean)
    batch.add_trial(errored)

    assert batch.grader_error_count == 1
    assert batch.grader_error_rate == 0.5


def test_runner_marks_grader_error_outcome() -> None:
    async def agent_fn(input_data: dict) -> dict:
        return {"ok": True}

    runner = EvaluationRunner(
        adapter=SimpleAdapter(agent_fn),
        graders=[_CrashingGrader()],
    )
    batch = asyncio.run(runner.run(EvalSet(name="s", tasks=[_task()])))

    assert batch.total_count == 1
    outcome = batch.trials[0].outcomes[0]
    assert outcome.grader_error is True
    assert batch.grader_error_count == 1


def test_runner_reraises_memory_error_from_grader() -> None:
    async def agent_fn(input_data: dict) -> dict:
        return {"ok": True}

    runner = EvaluationRunner(
        adapter=SimpleAdapter(agent_fn),
        graders=[_CrashingGrader(MemoryError("oom"))],
    )

    with pytest.raises(MemoryError):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task()])))


def test_report_surfaces_grader_error_stats() -> None:
    batch = TrialBatch()
    errored = Trial(task_id="t1")
    errored.add_outcome(_outcome(grader_error=True))
    batch.add_trial(errored)

    report = ReportGenerator().build_report(batch)

    assert report.grader_error_count == 1
    assert report.grader_error_rate == 1.0
    assert report.to_dict()["grader_error_count"] == 1
