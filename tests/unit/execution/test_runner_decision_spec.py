"""Tests for DecisionSpec wiring in the EvaluationRunner.

The runner stamps its configured DecisionSpec onto every transcript so
baselines can be compared with reproducibility fingerprints — without
each adapter having to remember to do it.
"""

import asyncio

from tracelens.core.decision_spec import AgentSpec, DecisionSpec
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.execution.agent_adapter import AgentAdapter, SimpleAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig


def _task(task_id: str = "t1") -> Task:
    return Task(task_id=task_id, name="t", description="d", input_data={"x": 1})


async def _agent_fn(input_data: dict) -> dict:
    return {"ok": True}


class _SpecSettingAdapter(AgentAdapter):
    """Adapter that stamps its own DecisionSpec on the transcript."""

    def __init__(self, spec: DecisionSpec) -> None:
        self._spec = spec

    async def run(self, task: Task) -> Transcript:
        transcript = Transcript(task_id=task.task_id)
        transcript.decision_spec = self._spec
        return transcript


def test_runner_assigns_decision_spec_to_transcripts() -> None:
    spec = DecisionSpec(agent=AgentSpec(agent_name="my_agent", agent_version="1.0"))
    runner = EvaluationRunner(
        adapter=SimpleAdapter(_agent_fn),
        graders=[],
        config=RunnerConfig(num_runs=2),
        decision_spec=spec,
    )

    batch = asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a"), _task("b")])))

    assert batch.total_count == 4
    for trial in batch.trials:
        assert trial.transcript is not None
        assert trial.transcript.decision_spec is not None
        assert trial.transcript.decision_spec.fingerprint == spec.fingerprint
        assert trial.fingerprint == spec.fingerprint


def test_runner_preserves_adapter_provided_spec() -> None:
    adapter_spec = DecisionSpec(agent=AgentSpec(agent_name="adapter_owned"))
    runner_spec = DecisionSpec(agent=AgentSpec(agent_name="runner_owned"))
    runner = EvaluationRunner(
        adapter=_SpecSettingAdapter(adapter_spec),
        graders=[],
        decision_spec=runner_spec,
    )

    batch = asyncio.run(runner.run(EvalSet(name="s", tasks=[_task()])))

    transcript = batch.trials[0].transcript
    assert transcript is not None
    assert transcript.decision_spec is not None
    assert transcript.decision_spec.fingerprint == adapter_spec.fingerprint


def test_runner_without_spec_leaves_transcript_spec_none() -> None:
    runner = EvaluationRunner(adapter=SimpleAdapter(_agent_fn), graders=[])

    batch = asyncio.run(runner.run(EvalSet(name="s", tasks=[_task()])))

    transcript = batch.trials[0].transcript
    assert transcript is not None
    assert transcript.decision_spec is None
