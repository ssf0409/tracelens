"""Tests for runner checkpointing and resume.

Long evals must survive crashes: trials are periodically persisted to a
checkpoint file, and a rerun with the same checkpoint path skips trials
that already completed. Infra-errored trials are NOT skipped — they carry
no signal about the agent, so a resume re-runs them. The checkpoint also
records the identity (eval-set hash + adapter/grader classes) of the run
that produced it, and resuming against a mismatched checkpoint refuses
instead of silently merging foreign trials.
"""

import asyncio
import json
import logging
from pathlib import Path

import pytest

from tracelens.core.grader import CodeGrader
from tracelens.core.task import EvalSet, Task
from tracelens.core.transcript import Transcript
from tracelens.core.trial import InfraError, Trial, TrialBatch, TrialStatus
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.runner import (
    CheckpointError,
    EvaluationRunner,
    RunnerConfig,
)


def _task(task_id: str, x: int = 1) -> Task:
    return Task(task_id=task_id, name="t", description="d", input_data={"x": x})


class _CountingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.run_calls: list[str] = []

    async def run(self, task: Task) -> Transcript:
        self.run_calls.append(task.task_id)
        return Transcript(task_id=task.task_id, final_output={"ok": True})


class _OtherAdapter(AgentAdapter):
    async def run(self, task: Task) -> Transcript:
        return Transcript(task_id=task.task_id, final_output={"ok": True})


class _SelectiveInfraAdapter(AgentAdapter):
    """Raises InfraError for the configured task ids, succeeds otherwise."""

    def __init__(self, fail_task_ids: set[str] | None = None) -> None:
        self.fail_task_ids = fail_task_ids or set()
        self.run_calls: list[str] = []

    async def run(self, task: Task) -> Transcript:
        self.run_calls.append(task.task_id)
        if task.task_id in self.fail_task_ids:
            raise InfraError("flaky infra")
        return Transcript(task_id=task.task_id, final_output={"ok": True})


class _GraderA(CodeGrader):
    def __init__(self) -> None:
        super().__init__("grader_a")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"a": 1.0}

    def determine_pass(
        self, metrics: dict[str, float], task: Task
    ) -> tuple[bool, float]:
        return True, 1.0


class _GraderB(CodeGrader):
    def __init__(self) -> None:
        super().__init__("grader_b")

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {"b": 1.0}

    def determine_pass(
        self, metrics: dict[str, float], task: Task
    ) -> tuple[bool, float]:
        return True, 1.0


def _runner(
    adapter: AgentAdapter,
    checkpoint: Path,
    graders: list[CodeGrader] | None = None,
) -> EvaluationRunner:
    return EvaluationRunner(
        adapter=adapter,
        graders=graders or [],
        config=RunnerConfig(checkpoint_path=str(checkpoint), checkpoint_interval=1),
    )


def test_checkpoint_written_at_interval(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    runner = _runner(_CountingAdapter(), checkpoint)

    asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a"), _task("b")])))

    assert checkpoint.exists()
    data = json.loads(checkpoint.read_text())
    loaded = TrialBatch.from_dict(data["batch"])
    assert loaded.total_count == 2
    assert all(t.is_complete for t in loaded.trials)


def test_checkpoint_records_run_identity(tmp_path: Path) -> None:
    """The checkpoint carries eval-set hash + adapter/grader identity."""
    checkpoint = tmp_path / "checkpoint.json"
    runner = _runner(_CountingAdapter(), checkpoint, graders=[_GraderA()])

    asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a")])))

    data = json.loads(checkpoint.read_text())
    assert data["version"] == 1
    identity = data["identity"]
    assert identity["eval_set_hash"]
    assert identity["adapter"].endswith("_CountingAdapter")
    assert any(g.endswith("_GraderA") for g in identity["graders"])


def test_resume_skips_completed_trials(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    eval_set = EvalSet(name="s", tasks=[_task("a"), _task("b")])

    first = _CountingAdapter()
    asyncio.run(_runner(first, checkpoint).run(eval_set))
    assert sorted(first.run_calls) == ["a", "b"]

    # Second run resumes from the checkpoint: nothing left to do.
    second = _CountingAdapter()
    batch = asyncio.run(_runner(second, checkpoint).run(eval_set))

    assert second.run_calls == []
    assert batch.total_count == 2


def test_resume_reruns_incomplete_trials(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"

    # Simulate a crash: task "a" completed, task "b" was mid-flight.
    # (Legacy bare-batch format — also exercises backward compatibility.)
    stale = TrialBatch()
    done = Trial(task_id="a", run_index=0, status=TrialStatus.COMPLETED)
    crashed = Trial(task_id="b", run_index=0, status=TrialStatus.RUNNING)
    stale.add_trial(done)
    stale.add_trial(crashed)
    checkpoint.write_text(json.dumps(stale.to_dict()))

    adapter = _CountingAdapter()
    batch = asyncio.run(
        _runner(adapter, checkpoint).run(
            EvalSet(name="s", tasks=[_task("a"), _task("b")])
        )
    )

    assert adapter.run_calls == ["b"]
    assert batch.total_count == 2


def test_resume_reruns_infra_errored_trials(tmp_path: Path) -> None:
    """INFRA_ERROR trials are not permanently skipped on resume."""
    checkpoint = tmp_path / "checkpoint.json"
    eval_set = EvalSet(name="s", tasks=[_task("a"), _task("b")])

    first = _SelectiveInfraAdapter(fail_task_ids={"a"})
    batch = asyncio.run(_runner(first, checkpoint).run(eval_set))
    assert batch.infra_error_count == 1

    # Infra recovered: the resume re-runs "a" but not the completed "b".
    second = _SelectiveInfraAdapter()
    batch = asyncio.run(_runner(second, checkpoint).run(eval_set))

    assert second.run_calls == ["a"]
    assert batch.total_count == 2
    assert batch.infra_error_count == 0
    statuses = {t.task_id: t.status for t in batch.trials}
    assert statuses == {"a": TrialStatus.COMPLETED, "b": TrialStatus.COMPLETED}


def test_resume_still_skips_timeout_trials(tmp_path: Path) -> None:
    """TIMEOUT is an observation about the agent, not infra — stays skipped."""
    checkpoint = tmp_path / "checkpoint.json"

    stale = TrialBatch()
    stale.add_trial(Trial(task_id="a", run_index=0, status=TrialStatus.TIMEOUT))
    stale.add_trial(Trial(task_id="b", run_index=0, status=TrialStatus.COMPLETED))
    checkpoint.write_text(json.dumps(stale.to_dict()))

    adapter = _CountingAdapter()
    batch = asyncio.run(
        _runner(adapter, checkpoint).run(
            EvalSet(name="s", tasks=[_task("a"), _task("b")])
        )
    )

    assert adapter.run_calls == []
    assert batch.total_count == 2
    statuses = {t.task_id: t.status for t in batch.trials}
    assert statuses["a"] == TrialStatus.TIMEOUT


def test_legacy_checkpoint_warns_but_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pre-identity checkpoints can't be validated: warn loudly, still load."""
    checkpoint = tmp_path / "checkpoint.json"

    stale = TrialBatch()
    stale.add_trial(Trial(task_id="a", run_index=0, status=TrialStatus.COMPLETED))
    checkpoint.write_text(json.dumps(stale.to_dict()))

    adapter = _CountingAdapter()
    with caplog.at_level(logging.WARNING):
        batch = asyncio.run(
            _runner(adapter, checkpoint).run(EvalSet(name="s", tasks=[_task("a")]))
        )

    assert adapter.run_calls == []
    assert batch.total_count == 1
    assert any("identity" in rec.message.lower() for rec in caplog.records)


def test_corrupt_checkpoint_raises_clear_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{not valid json")

    runner = _runner(_CountingAdapter(), checkpoint)
    with pytest.raises(CheckpointError, match="checkpoint"):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a")])))


def test_unparseable_checkpoint_schema_raises_clear_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"version": 1, "identity": {}, "batch": 42}))

    runner = _runner(_CountingAdapter(), checkpoint)
    with pytest.raises(CheckpointError, match="checkpoint"):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a")])))


def test_mismatched_eval_set_refuses_resume(tmp_path: Path) -> None:
    """Same task ids, different content: the checkpoint must not merge."""
    checkpoint = tmp_path / "checkpoint.json"

    asyncio.run(
        _runner(_CountingAdapter(), checkpoint).run(
            EvalSet(name="s", tasks=[_task("a", x=1)])
        )
    )

    runner = _runner(_CountingAdapter(), checkpoint)
    with pytest.raises(CheckpointError, match="eval set"):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a", x=2)])))


def test_malformed_identity_raises_clear_error(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {"version": 1, "identity": "garbage", "batch": TrialBatch().to_dict()}
        )
    )

    runner = _runner(_CountingAdapter(), checkpoint)
    with pytest.raises(CheckpointError, match="identity"):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a")])))


def test_mismatched_graders_refuse_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    eval_set = EvalSet(name="s", tasks=[_task("a")])

    asyncio.run(
        _runner(_CountingAdapter(), checkpoint, graders=[_GraderA()]).run(eval_set)
    )

    runner = _runner(_CountingAdapter(), checkpoint, graders=[_GraderB()])
    with pytest.raises(CheckpointError, match="graders"):
        asyncio.run(runner.run(eval_set))


def test_mismatched_adapter_refuses_resume(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    eval_set = EvalSet(name="s", tasks=[_task("a")])

    asyncio.run(_runner(_CountingAdapter(), checkpoint).run(eval_set))

    runner = _runner(_OtherAdapter(), checkpoint)
    with pytest.raises(CheckpointError, match="adapter"):
        asyncio.run(runner.run(eval_set))


def test_grader_order_change_still_resumes(tmp_path: Path) -> None:
    """Grader order doesn't change what was graded — no false refusal."""
    checkpoint = tmp_path / "checkpoint.json"
    eval_set = EvalSet(name="s", tasks=[_task("a")])

    asyncio.run(
        _runner(_CountingAdapter(), checkpoint, graders=[_GraderA(), _GraderB()]).run(
            eval_set
        )
    )

    second = _CountingAdapter()
    batch = asyncio.run(
        _runner(second, checkpoint, graders=[_GraderB(), _GraderA()]).run(eval_set)
    )

    assert second.run_calls == []
    assert batch.total_count == 1


def test_no_checkpoint_config_writes_nothing(tmp_path: Path) -> None:
    runner = EvaluationRunner(adapter=_CountingAdapter(), graders=[])
    asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("a")])))
    assert list(tmp_path.iterdir()) == []


def test_undecodable_checkpoint_raises_checkpoint_error(tmp_path: Path) -> None:
    """Binary/truncated-multibyte corruption must raise CheckpointError,
    not leak UnicodeDecodeError."""
    ckpt = tmp_path / "c.json"
    ckpt.write_bytes(b"\xff\xfe\x00garbage\x80")

    runner = _runner(_CountingAdapter(), ckpt)
    with pytest.raises(CheckpointError, match="[Cc]orrupt"):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("t1")])))


def test_envelope_without_identity_is_corrupt(tmp_path: Path) -> None:
    """A versioned envelope must carry its identity — a missing one is
    corruption, not a legacy file (legacy files are bare batches)."""
    ckpt = tmp_path / "c.json"
    ckpt.write_text(json.dumps({"version": 1, "batch": TrialBatch().to_dict()}))

    runner = _runner(_CountingAdapter(), ckpt)
    with pytest.raises(CheckpointError, match="identity"):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("t1")])))


def test_unknown_checkpoint_version_raises(tmp_path: Path) -> None:
    ckpt = tmp_path / "c.json"
    ckpt.write_text(json.dumps({
        "version": 99,
        "identity": {"eval_set_hash": "x", "adapter": "a", "graders": []},
        "batch": TrialBatch().to_dict(),
    }))

    runner = _runner(_CountingAdapter(), ckpt)
    with pytest.raises(CheckpointError, match="version"):
        asyncio.run(runner.run(EvalSet(name="s", tasks=[_task("t1")])))


def test_decision_spec_change_refuses_resume(tmp_path: Path) -> None:
    """The DecisionSpec is the run's reproducibility identity — resuming a
    checkpoint recorded under a different spec would mix two configurations."""
    from tracelens.core.decision_spec import DecisionSpec, InfraConfig

    ckpt = tmp_path / "c.json"
    eval_set = EvalSet(name="s", tasks=[_task("t1")])
    spec_a = DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=2048))
    spec_b = DecisionSpec(infra=InfraConfig(memory_hard_limit_mb=512))

    runner = EvaluationRunner(
        _CountingAdapter(),
        [_GraderA()],
        RunnerConfig(checkpoint_path=str(ckpt)),
        decision_spec=spec_a,
    )
    asyncio.run(runner.run(eval_set))

    # Same spec resumes fine.
    same = EvaluationRunner(
        _CountingAdapter(),
        [_GraderA()],
        RunnerConfig(checkpoint_path=str(ckpt)),
        decision_spec=spec_a,
    )
    asyncio.run(same.run(eval_set))

    # Different spec refuses.
    different = EvaluationRunner(
        _CountingAdapter(),
        [_GraderA()],
        RunnerConfig(checkpoint_path=str(ckpt)),
        decision_spec=spec_b,
    )
    with pytest.raises(CheckpointError, match="decision spec"):
        asyncio.run(different.run(eval_set))


def test_resume_reruns_skipped_trials(tmp_path: Path) -> None:
    """SKIPPED trials carry no signal (e.g. fail-fast placeholders from a
    prior run) — resume must re-run them, not load them as done."""
    ckpt = tmp_path / "c.json"
    eval_set = EvalSet(name="s", tasks=[_task("t1")])

    runner = _runner(_CountingAdapter(), ckpt)
    asyncio.run(runner.run(eval_set))

    # Doctor the checkpoint: mark the completed trial SKIPPED.
    data = json.loads(ckpt.read_text())
    data["batch"]["trials"][0]["status"] = "skipped"
    data["batch"]["trials"][0]["outcomes"] = []
    ckpt.write_text(json.dumps(data))

    adapter = _CountingAdapter()
    resumed = _runner(adapter, ckpt)
    batch = asyncio.run(resumed.run(eval_set))

    assert adapter.run_calls == ["t1"]  # re-ran, not skipped
    assert all(t.status != TrialStatus.SKIPPED for t in batch.trials)
