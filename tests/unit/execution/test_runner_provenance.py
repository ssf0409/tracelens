"""The runner records provenance on every batch (issue #51)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from tracelens.core.decision_spec import DecisionSpec, ModelConfig
from tracelens.core.grader import CodeGrader
from tracelens.core.provenance import eval_set_hash, task_content_hash
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import TrialBatch
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.runner import EvaluationRunner, RunnerConfig


class _Echo(AgentAdapter):
    async def run(self, task: Task) -> Transcript:
        return Transcript(task_id=task.task_id, final_output=dict(task.input_data))


class _Pass(CodeGrader):
    def __init__(self) -> None:
        super().__init__("always")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"ok": 1.0}

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return True, 1.0


def _suite() -> EvalSet:
    return EvalSet(name="suite", tasks=[
        Task(task_id="a", name="a", input_data={"x": 1}),
        Task(task_id="b", name="b", input_data={"x": 2}),
    ])


def test_run_records_provenance_and_checkpoint_identity_agrees(tmp_path: Path) -> None:
    checkpoint = tmp_path / "cp.json"
    spec = DecisionSpec(model=ModelConfig(provider="p", model_id="m"))
    runner = EvaluationRunner(
        _Echo(), [_Pass()],
        RunnerConfig(num_runs=2, checkpoint_path=str(checkpoint)),
        decision_spec=spec,
    )
    batch = asyncio.run(runner.run(_suite()))

    prov = batch.provenance
    assert prov is not None
    assert prov.run_id == batch.batch_id
    assert prov.started_at == batch.started_at and prov.completed_at == batch.completed_at
    assert prov.measurement.eval_set_name == "suite"
    assert prov.measurement.eval_set_hash == eval_set_hash(_suite())
    assert prov.measurement.task_hashes == {
        t.task_id: task_content_hash(t) for t in _suite().tasks
    }
    assert [g.name for g in prov.measurement.graders] == ["always"]
    assert prov.measurement.runner.num_runs == 2
    assert prov.candidate.adapter.class_path.endswith("_Echo")
    assert prov.candidate.decision_spec_fingerprint == spec.fingerprint

    # One hashing rule: the checkpoint identity is derived from the same record.
    identity = json.loads(checkpoint.read_text())["identity"]
    assert identity == {
        "eval_set_hash": prov.measurement.eval_set_hash,
        "adapter": prov.candidate.adapter.class_path,
        "graders": [g.class_path for g in prov.measurement.graders],
        "decision_spec_fingerprint": spec.fingerprint,
    }


def test_provenance_survives_trials_json_round_trip_and_legacy_files_have_none() -> None:
    batch = asyncio.run(EvaluationRunner(_Echo(), [_Pass()]).run(_suite()))
    data = json.loads(json.dumps(batch.to_dict()))
    assert TrialBatch.from_dict(data).provenance == batch.provenance
    del data["provenance"]
    assert TrialBatch.from_dict(data).provenance is None


def test_unknown_schema_version_in_trials_file_is_rejected() -> None:
    batch = asyncio.run(EvaluationRunner(_Echo(), [_Pass()]).run(_suite()))
    data = batch.to_dict()
    data["provenance"]["schema_version"] = 2
    with pytest.raises(ValidationError, match="unknown provenance schema version 2"):
        TrialBatch.from_dict(data)
