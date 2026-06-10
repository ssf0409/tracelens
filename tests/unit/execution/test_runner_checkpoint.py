"""Tests for runner checkpointing and resume.

Long evals must survive crashes: trials are periodically persisted to a
checkpoint file, and a rerun with the same checkpoint path skips trials
that already completed.
"""

import asyncio
import json
from pathlib import Path

from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import Trial, TrialBatch, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig


def _task(task_id: str) -> Task:
    return Task(task_id=task_id, name="t", description="d", input_data={"x": 1})


class _CountingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.run_calls: list[str] = []

    async def run(self, task: Task) -> Transcript:
        self.run_calls.append(task.task_id)
        return Transcript(task_id=task.task_id, final_output={"ok": True})


def test_checkpoint_written_at_interval(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    runner = EvaluationRunner(
        adapter=_CountingAdapter(),
        graders=[],
        config=RunnerConfig(
            checkpoint_path=str(checkpoint), checkpoint_interval=1
        ),
    )

    asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a"), _task("b")])))

    assert checkpoint.exists()
    loaded = TrialBatch.from_dict(json.loads(checkpoint.read_text()))
    assert loaded.total_count == 2
    assert all(t.is_complete for t in loaded.trials)


def test_resume_skips_completed_trials(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    eval_set = EvalSet(name="s", tasks=[_task("a"), _task("b")])

    first = _CountingAdapter()
    runner = EvaluationRunner(
        adapter=first,
        graders=[],
        config=RunnerConfig(checkpoint_path=str(checkpoint), checkpoint_interval=1),
    )
    asyncio.run(runner.run(eval_set))
    assert sorted(first.run_calls) == ["a", "b"]

    # Second run resumes from the checkpoint: nothing left to do.
    second = _CountingAdapter()
    runner = EvaluationRunner(
        adapter=second,
        graders=[],
        config=RunnerConfig(checkpoint_path=str(checkpoint), checkpoint_interval=1),
    )
    batch = asyncio.run(runner.run(eval_set))

    assert second.run_calls == []
    assert batch.total_count == 2


def test_resume_reruns_incomplete_trials(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"

    # Simulate a crash: task "a" completed, task "b" was mid-flight.
    stale = TrialBatch()
    done = Trial(task_id="a", run_index=0, status=TrialStatus.COMPLETED)
    crashed = Trial(task_id="b", run_index=0, status=TrialStatus.RUNNING)
    stale.add_trial(done)
    stale.add_trial(crashed)
    checkpoint.write_text(json.dumps(stale.to_dict()))

    adapter = _CountingAdapter()
    runner = EvaluationRunner(
        adapter=adapter,
        graders=[],
        config=RunnerConfig(checkpoint_path=str(checkpoint), checkpoint_interval=1),
    )
    batch = asyncio.run(
        runner.run(EvalSet(name="s", tasks=[_task("a"), _task("b")]))
    )

    assert adapter.run_calls == ["b"]
    assert batch.total_count == 2


def test_no_checkpoint_config_writes_nothing(tmp_path: Path) -> None:
    runner = EvaluationRunner(adapter=_CountingAdapter(), graders=[])
    asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a")])))
    assert list(tmp_path.iterdir()) == []
