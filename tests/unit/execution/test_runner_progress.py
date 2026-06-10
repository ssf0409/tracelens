"""Tests for runner progress reporting."""

import asyncio

from tracelens.core.task import EvalSet, Task
from tracelens.execution.agent_adapter import SimpleAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig


def _task(task_id: str) -> Task:
    return Task(task_id=task_id, name="t", description="d", input_data={"x": 1})


async def _agent_fn(input_data: dict) -> dict:
    return {"ok": True}


def test_progress_callback_called_once_per_trial() -> None:
    calls: list[tuple[int, int]] = []

    runner = EvaluationRunner(
        adapter=SimpleAdapter(_agent_fn),
        graders=[],
        config=RunnerConfig(
            num_runs=3,
            progress_callback=lambda done, total: calls.append((done, total)),
        ),
    )
    asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a"), _task("b")])))

    assert len(calls) == 6
    assert all(total == 6 for _, total in calls)
    assert [done for done, _ in calls] == [1, 2, 3, 4, 5, 6]


def test_no_callback_is_fine() -> None:
    runner = EvaluationRunner(
        adapter=SimpleAdapter(_agent_fn),
        graders=[],
        config=RunnerConfig(),
    )
    batch = asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a")])))
    assert batch.total_count == 1
