"""Tests for run provenance and comparison compatibility (issue #51)."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from tracelens.core.decision_spec import DecisionSpec, ModelConfig, PromptSpec
from tracelens.core.grader import CodeGrader
from tracelens.core.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    Compatibility,
    CompatibilityReport,
    ComponentIdentity,
    RunnerSettings,
    RunProvenance,
    build_provenance,
    canonical_json,
    check_compatibility,
    content_hash,
    eval_set_hash,
    task_content_hash,
)
from tracelens.core.task import EvalSet, Task, TaskExpectation
from tracelens.core.transcript import Transcript
from tracelens.execution.agent_adapter import AgentAdapter
from tracelens.execution.runner import RunnerConfig


class _Adapter(AgentAdapter):
    async def run(self, task: Task) -> Transcript:  # pragma: no cover - identity only
        return Transcript(task_id=task.task_id, final_output={})


class _VersionedAdapter(_Adapter):
    provenance_version = "agent-2.3.0"


class _Grader(CodeGrader):
    def __init__(self, grader_id: str = "g") -> None:
        super().__init__(grader_id)

    def compute_metrics(self, transcript: Transcript, task: Task) -> dict[str, float]:
        return {}  # pragma: no cover - identity only

    def determine_pass(self, metrics: dict[str, float], task: Task) -> tuple[bool, float]:
        return True, 1.0  # pragma: no cover - identity only


class _VersionedGrader(_Grader):
    provenance_version = "rubric-v4"


def _task(task_id: str, x: int = 1, **kwargs: Any) -> Task:
    return Task(task_id=task_id, name=f"task {task_id}", input_data={"x": x}, **kwargs)


def _settings(**overrides: Any) -> RunnerSettings:
    base: dict[str, Any] = {
        "num_runs": 2, "max_concurrency": 3, "timeout_seconds": 30.0,
        "max_infra_retries": 1, "infra_exception_types": ["builtins.ConnectionError"],
    }
    base.update(overrides)
    return RunnerSettings(**base)


def _provenance(
    tasks: list[Task],
    *,
    graders: list[Any] | None = None,
    adapter: Any = None,
    spec: DecisionSpec | None = None,
    settings: RunnerSettings | None = None,
    run_id: str = "run-1",
) -> RunProvenance:
    return build_provenance(
        eval_set=EvalSet(name="suite", tasks=tasks),
        adapter=adapter or _Adapter(),
        graders=graders if graders is not None else [_Grader()],
        settings=settings or _settings(),
        decision_spec=spec,
        run_id=run_id,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class TestHashing:
    def test_canonical_json_sorts_keys_and_coerces_unknown_types(self):
        assert canonical_json({"b": 1, "a": {"d": 2, "c": 3}}) == '{"a": {"c": 3, "d": 2}, "b": 1}'
        assert canonical_json({"when": datetime(2026, 1, 1)}) == '{"when": "2026-01-01 00:00:00"}'

    def test_content_hash_is_sha256_of_canonical_json(self):
        value = {"z": [1, 2], "a": "x"}
        expected = hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert content_hash(value) == expected
        assert content_hash({"a": "x", "z": [1, 2]}) == expected  # key order is irrelevant

    def test_task_hash_covers_every_field(self):
        base = _task("t", 1)
        variants = [
            _task("t", 2),
            _task("u", 1),
            Task(task_id="t", name="other name", input_data={"x": 1}),
            _task("t", 1, description="d"),
            _task("t", 1, expectation=TaskExpectation(expected_output="42")),
            _task("t", 1, metadata={"k": "v"}),
            _task("t", 1, tags=["a"]),
            _task("t", 1, difficulty="hard"),
            _task("t", 1, category="math"),
            _task("t", 1, timeout_seconds=1.0),
        ]
        hashes = {task_content_hash(v) for v in variants}
        assert task_content_hash(base) not in hashes
        assert len(hashes) == len(variants)
        assert task_content_hash(base) == task_content_hash(_task("t", 1))

    def test_eval_set_hash_ignores_task_order_and_suite_metadata(self):
        a, b = _task("a"), _task("b")
        assert eval_set_hash(EvalSet(name="x", tasks=[a, b])) == eval_set_hash(
            EvalSet(name="y", version="9.9.9", tasks=[b, a])
        )
        assert eval_set_hash(EvalSet(name="x", tasks=[a])) != eval_set_hash(
            EvalSet(name="x", tasks=[a, b])
        )

    def test_eval_set_hash_is_the_sorted_task_dump_hash(self):
        """Independent derivation of the documented rule (also the checkpoint identity)."""
        first, second = _task("b"), _task("a")
        dumps = sorted(
            (t.model_dump(mode="json") for t in (first, second)), key=lambda d: d["task_id"]
        )
        expected = hashlib.sha256(
            json.dumps(dumps, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert eval_set_hash(EvalSet(name="s", tasks=[first, second])) == expected


class TestIdentities:
    def test_component_identity_reads_declared_version_only(self):
        assert ComponentIdentity.of(_Adapter()) == ComponentIdentity(
            class_path=f"{__name__}._Adapter"
        )
        assert ComponentIdentity.of(_VersionedAdapter()).version == "agent-2.3.0"
        grader = ComponentIdentity.of(_VersionedGrader("quality"), name="quality")
        assert grader.describe() == f"quality ({__name__}._VersionedGrader) @ rubric-v4"

    def test_non_string_version_is_ignored(self):
        adapter = _Adapter()
        adapter.provenance_version = 3  # type: ignore[attr-defined]
        assert ComponentIdentity.of(adapter).version is None

    def test_runner_settings_from_config(self):
        config = RunnerConfig(
            num_runs=4, max_concurrency=2, timeout_seconds=12.5, max_infra_retries=3,
            infra_exception_types=(ConnectionError, MemoryError),
        )
        assert RunnerSettings.from_config(config) == RunnerSettings(
            num_runs=4, max_concurrency=2, timeout_seconds=12.5, max_infra_retries=3,
            infra_exception_types=["builtins.ConnectionError", "builtins.MemoryError"],
        )


class TestRunProvenance:
    def test_build_records_measurement_and_candidate(self):
        spec = DecisionSpec(model=ModelConfig(provider="p", model_id="m"))
        tasks = [_task("a"), _task("b")]
        prov = _provenance(
            tasks, graders=[_Grader("g1"), _VersionedGrader("g2")],
            adapter=_VersionedAdapter(), spec=spec,
        )
        assert prov.schema_version == PROVENANCE_SCHEMA_VERSION
        assert prov.run_id == "run-1" and prov.tracelens_version
        assert prov.measurement.eval_set_name == "suite"
        assert prov.measurement.task_hashes == {
            "a": task_content_hash(_task("a")), "b": task_content_hash(_task("b")),
        }
        assert prov.measurement.eval_set_hash == eval_set_hash(EvalSet(name="s", tasks=tasks))
        assert [g.name for g in prov.measurement.graders] == ["g1", "g2"]
        assert prov.measurement.graders[1].version == "rubric-v4"
        assert prov.measurement.runner == _settings()
        assert prov.candidate.adapter.version == "agent-2.3.0"
        assert prov.candidate.decision_spec_fingerprint == spec.fingerprint
        assert prov.candidate.decision_spec == spec
        assert prov.completed_at is None

    def test_json_round_trip_preserves_everything(self):
        spec = DecisionSpec(model=ModelConfig(provider="p", model_id="m"))
        prov = _provenance([_task("a")], spec=spec)
        prov.completed_at = datetime(2026, 1, 1, 0, 5, tzinfo=UTC)
        data = json.loads(json.dumps(prov.model_dump(mode="json")))
        assert RunProvenance.model_validate(data) == prov

    def test_unknown_schema_version_is_rejected_clearly(self):
        data = _provenance([_task("a")]).model_dump(mode="json")
        data["schema_version"] = 99
        with pytest.raises(ValidationError, match="unknown provenance schema version 99"):
            RunProvenance.model_validate(data)

    def test_summary_lines_name_the_essentials(self):
        spec = DecisionSpec(model=ModelConfig(provider="p", model_id="m"))
        prov = _provenance([_task("a")], spec=spec)
        text = "\n".join(prov.summary_lines())
        assert "Run: run-1" in text
        assert "Eval set: suite, 1 task(s), content " in text
        assert "Graders: g (" in text
        assert "Runner: 2 run(s) per task, timeout 30s, 1 infra retries" in text
        assert f"Candidate spec: {spec.fingerprint[:12]}" in text
        assert "Candidate spec: none declared" in "\n".join(_provenance([]).summary_lines())

    def test_no_prompt_text_by_default(self):
        spec = DecisionSpec(prompts=PromptSpec.from_prompts(system_prompt="SECRET SYSTEM PROMPT"))
        dumped = json.dumps(_provenance([_task("a")], spec=spec).model_dump(mode="json"))
        assert "SECRET SYSTEM PROMPT" not in dumped
        assert spec.prompts is not None and spec.prompts.system_prompt_hash in dumped


class TestCompatibility:
    def test_identical_runs_are_compatible_and_candidate_unchanged(self):
        a = _provenance([_task("a"), _task("b")])
        b = _provenance([_task("a"), _task("b")], run_id="run-2")
        report = check_compatibility(a, b)
        assert report.status is Compatibility.COMPATIBLE and report.compatible
        assert report.tasks is not None
        assert report.tasks.same == ["a", "b"] and not report.tasks.changed
        assert report.graders_changed is False and report.candidate_changed is False
        assert report.reasons == [] and report.notes == []
        assert report.summary_line() == (
            "Measurement compatibility: compatible; 2 shared task(s), same graders; "
            "candidate unchanged"
        )

    def test_reordered_tasks_are_compatible(self):
        """Independent expectation: the same tasks in another order measure the same thing."""
        a = _provenance([_task("a"), _task("b")])
        b = _provenance([_task("b"), _task("a")])
        assert check_compatibility(a, b).status is Compatibility.COMPATIBLE
        assert a.measurement.eval_set_hash == b.measurement.eval_set_hash

    def test_changed_task_content_is_incompatible_and_named(self):
        a = _provenance([_task("a", 1), _task("b")])
        b = _provenance([_task("a", 2), _task("b")])
        report = check_compatibility(a, b)
        assert report.status is Compatibility.INCOMPATIBLE and not report.compatible
        assert report.tasks is not None
        assert report.tasks.changed == ["a"] and report.tasks.same == ["b"]
        assert report.reasons == ["1 task(s) share an id but differ in content: a"]
        assert report.summary_line() == (
            "Measurement compatibility: incompatible; 1 task(s) share an id but differ "
            "in content: a"
        )

    def test_added_and_removed_tasks_are_incompatible(self):
        a = _provenance([_task("a"), _task("b")])
        b = _provenance([_task("b"), _task("c")])
        report = check_compatibility(a, b)
        assert report.status is Compatibility.INCOMPATIBLE
        assert report.tasks is not None
        assert report.tasks.only_in_a == ["a"] and report.tasks.only_in_b == ["c"]
        assert report.reasons == ["1 task(s) only in A: a", "1 task(s) only in B: c"]

    def test_long_id_lists_are_truncated(self):
        a = _provenance([_task(f"t{i}") for i in range(12)])
        report = check_compatibility(a, _provenance([]))
        assert report.reasons == [
            "12 task(s) only in A: t0, t1, t10, t11, t2, t3, t4, t5, and 4 more"
        ]

    def test_changed_grader_class_or_version_is_incompatible(self):
        a = _provenance([_task("a")], graders=[_Grader("g")])
        b = _provenance([_task("a")], graders=[_VersionedGrader("g")])
        report = check_compatibility(a, b)
        assert report.status is Compatibility.INCOMPATIBLE and report.graders_changed
        assert report.reasons[0].startswith("graders differ: A = [g (")

    def test_runner_settings_and_version_differences_are_notes(self):
        a = _provenance([_task("a")], settings=_settings(num_runs=2, timeout_seconds=30.0))
        b = _provenance([_task("a")], settings=_settings(num_runs=5, timeout_seconds=60.0))
        b.tracelens_version = "0.0.1"
        report = check_compatibility(a, b)
        assert report.status is Compatibility.COMPATIBLE
        assert report.notes == [
            "runner num_runs differs (2 vs 5)",
            "runner timeout_seconds differs (30.0 vs 60.0)",
            f"TraceLens version differs ({a.tracelens_version} vs 0.0.1)",
        ]
        assert "; note: runner num_runs differs (2 vs 5)" in report.summary_line()

    def test_changed_candidate_is_compatible_and_reported_with_a_diff(self):
        spec_a = DecisionSpec(model=ModelConfig(provider="p", model_id="m1"))
        spec_b = DecisionSpec(model=ModelConfig(provider="p", model_id="m2"))
        report = check_compatibility(
            _provenance([_task("a")], spec=spec_a), _provenance([_task("a")], spec=spec_b)
        )
        assert report.status is Compatibility.COMPATIBLE
        assert report.candidate_changed is True and report.adapter_changed is False
        assert list(report.candidate_diff) == ["model"]
        assert report.candidate_diff["model"][0]["model_id"] == "m1"
        assert report.candidate_diff["model"][1]["model_id"] == "m2"
        assert report.summary_line().endswith("candidate changed (model)")

    def test_changed_adapter_is_a_candidate_change(self):
        report = check_compatibility(
            _provenance([_task("a")], adapter=_Adapter()),
            _provenance([_task("a")], adapter=_VersionedAdapter()),
        )
        assert report.compatible and report.adapter_changed and report.candidate_changed

    def test_spec_on_one_side_only_is_noted(self):
        report = check_compatibility(
            _provenance([_task("a")]), _provenance([_task("a")], spec=DecisionSpec())
        )
        assert report.candidate_changed is True and report.candidate_diff == {}
        assert report.notes == ["a DecisionSpec is declared on one side only; no candidate diff"]

    @pytest.mark.parametrize("which", ["a", "b", "both"])
    def test_missing_provenance_is_unknown_never_a_silent_match(self, which):
        prov = _provenance([_task("a")])
        a = None if which in ("a", "both") else prov
        b = None if which in ("b", "both") else prov
        report = check_compatibility(a, b)
        assert report.status is Compatibility.UNKNOWN and not report.compatible
        assert report.tasks is None and report.candidate_changed is None
        labels = {"a": ["A"], "b": ["B"], "both": ["A", "B"]}[which]
        assert [reason.split(" ")[1] for reason in report.reasons] == labels
        assert report.summary_line().startswith("Measurement compatibility: unknown; run ")

    def test_hash_only_difference_without_task_detail_is_incompatible(self):
        a = _provenance([_task("a")])
        b = _provenance([_task("a")])
        a.measurement.task_hashes = {}
        b.measurement.task_hashes = {}
        b.measurement.eval_set_hash = "f" * 64
        report = check_compatibility(a, b)
        assert report.status is Compatibility.INCOMPATIBLE
        assert "no per-task detail is available" in report.reasons[0]

    def test_report_round_trips_through_json(self):
        report = check_compatibility(_provenance([_task("a", 1)]), _provenance([_task("a", 2)]))
        assert CompatibilityReport.model_validate(json.loads(report.model_dump_json())) == report
