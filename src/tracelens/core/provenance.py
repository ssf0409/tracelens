"""Versioned run provenance: what was measured, with what, on which candidate.

A :class:`RunProvenance` envelope is recorded by ``EvaluationRunner`` on
every ``TrialBatch`` and carried into ``ReportData``, so a saved artifact
says which task content, graders, and runner settings produced its numbers
and which candidate (adapter plus ``DecisionSpec``) was under test.

Two sides are kept apart on purpose:

* ``measurement`` is the instrument: eval-set content (a hash per task),
  grader identities, and runner settings. Two runs are *comparable* only
  when this side matches; a difference here means the numbers measure
  different things, however similar the task ids look.
* ``candidate`` is the thing under test: the adapter identity and the
  ``DecisionSpec`` fingerprint. This side is *expected* to differ between
  the two runs of a comparison; recording it lets a report say what
  changed. That is attribution evidence, never proof that two runs executed
  identical code or that a change caused an outcome difference.

Hashing rule, shared with checkpoint identity: a value is serialized as JSON
with sorted keys and ``str`` coercion for unknown types, then SHA-256
hashed. A task's content hash covers every ``Task`` field (id, name,
description, input, expectation, metadata, tags, difficulty, category,
timeout), so two tasks with the same id but different content never
collide. The eval-set hash covers the tasks sorted by id, so task order
never matters and eval-set-level fields (name, version, timestamps) do not
count.

Not recorded: credentials, full prompts (``DecisionSpec`` stores prompt
hashes unless the user opted into storing text), and adapter or grader
object state. Identities are declared class paths plus an optional
``provenance_version`` string attribute a component may define.

Artifacts written before provenance existed load with ``provenance=None``;
nothing is invented for them, and comparisons report the compatibility of
such runs as *unknown*. An artifact whose ``schema_version`` this TraceLens
does not know is rejected with a clear error.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from tracelens._version import __version__
from tracelens.core.decision_spec import DecisionSpec
from tracelens.core.task import EvalSet, Task

if TYPE_CHECKING:
    from tracelens.execution.runner import RunnerConfig

PROVENANCE_SCHEMA_VERSION = 1
_SHORT = 12


# --- Canonical hashing -------------------------------------------------------


def canonical_json(value: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, ``str`` for unknown types.

    This is the one serialization rule behind every content hash TraceLens
    records (checkpoint identity, task content, eval-set content).
    """
    return json.dumps(value, sort_keys=True, default=str)


def content_hash(value: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def task_content_hash(task: Task) -> str:
    """Content identity of one task: every field, including ``task_id``."""
    return content_hash(task.model_dump(mode="json"))


def eval_set_hash(eval_set: EvalSet) -> str:
    """Content identity of an eval set's tasks, independent of task order.

    The tasks' JSON dumps are sorted by task id and hashed together; this is
    also the checkpoint identity hash. EvalSet-level fields (name, version,
    timestamps) are excluded so renaming a suite does not change what it
    measures.
    """
    tasks = sorted(
        (t.model_dump(mode="json") for t in eval_set.tasks),
        key=lambda d: str(d.get("task_id")),
    )
    return content_hash(tasks)


def class_path(obj: object) -> str:
    """Dotted import path of an object's class."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def short_hash(digest: str | None) -> str:
    """The first characters of a hash for human-readable output."""
    return digest[:_SHORT] if digest else "n/a"


# --- The envelope ------------------------------------------------------------


class ComponentIdentity(BaseModel):
    """Declared identity of an adapter or grader.

    ``version`` is the component's ``provenance_version`` attribute when it
    declares one; TraceLens never serializes object state.
    """

    class_path: str
    name: str | None = None
    version: str | None = None

    @classmethod
    def of(cls, obj: object, *, name: str | None = None) -> ComponentIdentity:
        """Identity of a live component, reading its declared version if any."""
        declared = getattr(obj, "provenance_version", None)
        return cls(
            class_path=class_path(obj),
            name=name,
            version=declared if isinstance(declared, str) and declared else None,
        )

    def describe(self) -> str:
        text = self.class_path
        if self.name:
            text = f"{self.name} ({text})"
        if self.version:
            text += f" @ {self.version}"
        return text


class RunnerSettings(BaseModel):
    """Runner settings that shape how outcomes were measured."""

    num_runs: int
    max_concurrency: int
    timeout_seconds: float
    max_infra_retries: int
    infra_exception_types: list[str] = Field(default_factory=list)

    @classmethod
    def from_config(cls, config: RunnerConfig) -> RunnerSettings:
        return cls(
            num_runs=config.num_runs,
            max_concurrency=config.max_concurrency,
            timeout_seconds=config.timeout_seconds,
            max_infra_retries=config.max_infra_retries,
            infra_exception_types=[
                f"{t.__module__}.{t.__qualname__}" for t in config.infra_exception_types
            ],
        )


class MeasurementSetup(BaseModel):
    """The instrument: task content, graders, and runner settings."""

    eval_set_name: str | None = None
    eval_set_hash: str
    task_hashes: dict[str, str] = Field(default_factory=dict)
    graders: list[ComponentIdentity] = Field(default_factory=list)
    runner: RunnerSettings


class CandidateSpec(BaseModel):
    """The thing under test: adapter identity and the declared DecisionSpec."""

    adapter: ComponentIdentity
    decision_spec_fingerprint: str | None = None
    decision_spec: DecisionSpec | None = None


class RunProvenance(BaseModel):
    """Versioned envelope describing one run's measurement and candidate."""

    schema_version: int = PROVENANCE_SCHEMA_VERSION
    run_id: str
    tracelens_version: str = __version__
    started_at: datetime | None = None
    completed_at: datetime | None = None
    measurement: MeasurementSetup
    candidate: CandidateSpec

    @field_validator("schema_version")
    @classmethod
    def _known_schema(cls, value: int) -> int:
        if value != PROVENANCE_SCHEMA_VERSION:
            raise ValueError(
                f"unknown provenance schema version {value!r}; this TraceLens "
                f"({__version__}) reads version {PROVENANCE_SCHEMA_VERSION}"
            )
        return value

    def summary_lines(self) -> list[str]:
        """Human-readable ``label: value`` lines for reports."""
        m, c = self.measurement, self.candidate
        suite = m.eval_set_name or "(unnamed)"
        lines = [
            f"Run: {self.run_id} (TraceLens {self.tracelens_version})",
            (
                f"Eval set: {suite}, {len(m.task_hashes)} task(s), "
                f"content {short_hash(m.eval_set_hash)}"
            ),
            "Graders: " + (", ".join(g.describe() for g in m.graders) or "none"),
            (
                f"Runner: {m.runner.num_runs} run(s) per task, "
                f"timeout {m.runner.timeout_seconds:g}s, "
                f"{m.runner.max_infra_retries} infra retries"
            ),
            f"Adapter: {c.adapter.describe()}",
            (
                f"Candidate spec: {short_hash(c.decision_spec_fingerprint)}"
                if c.decision_spec_fingerprint
                else "Candidate spec: none declared"
            ),
        ]
        if self.started_at and self.completed_at:
            lines.append(
                f"Ran: {self.started_at.isoformat()} to {self.completed_at.isoformat()}"
            )
        return lines


def build_provenance(
    *,
    eval_set: EvalSet,
    adapter: object,
    graders: Sequence[object],
    settings: RunnerSettings,
    decision_spec: DecisionSpec | None,
    run_id: str,
    started_at: datetime | None,
) -> RunProvenance:
    """Record the provenance of a run about to execute."""
    return RunProvenance(
        run_id=run_id,
        started_at=started_at,
        measurement=MeasurementSetup(
            eval_set_name=eval_set.name,
            eval_set_hash=eval_set_hash(eval_set),
            task_hashes={t.task_id: task_content_hash(t) for t in eval_set.tasks},
            graders=[
                ComponentIdentity.of(g, name=getattr(g, "grader_id", None))
                for g in graders
            ],
            runner=settings,
        ),
        candidate=CandidateSpec(
            adapter=ComponentIdentity.of(adapter),
            decision_spec_fingerprint=(
                decision_spec.fingerprint if decision_spec is not None else None
            ),
            decision_spec=decision_spec,
        ),
    )


# --- Compatibility -----------------------------------------------------------


class Compatibility(StrEnum):
    """Whether two runs measured the same thing."""

    COMPATIBLE = "compatible"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class TaskAlignment(BaseModel):
    """How two runs' task sets line up, by id and by content."""

    same: list[str] = Field(default_factory=list)
    changed: list[str] = Field(default_factory=list)
    only_in_a: list[str] = Field(default_factory=list)
    only_in_b: list[str] = Field(default_factory=list)

    @classmethod
    def between(cls, a: Mapping[str, str], b: Mapping[str, str]) -> TaskAlignment:
        return cls(
            same=sorted(t for t in a if t in b and a[t] == b[t]),
            changed=sorted(t for t in a if t in b and a[t] != b[t]),
            only_in_a=sorted(t for t in a if t not in b),
            only_in_b=sorted(t for t in b if t not in a),
        )


class CompatibilityReport(BaseModel):
    """Whether two runs measure the same thing, and what differs.

    ``status`` answers the measurement question only. ``candidate_changed``
    and ``candidate_diff`` describe the thing under test, which is expected
    to differ between the two sides of a comparison; they are attribution
    evidence, never proof of what caused an outcome change.
    """

    status: Compatibility
    reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tasks: TaskAlignment | None = None
    graders_changed: bool | None = None
    adapter_changed: bool | None = None
    candidate_changed: bool | None = None
    candidate_diff: dict[str, list[Any]] = Field(default_factory=dict)

    @property
    def compatible(self) -> bool:
        return self.status is Compatibility.COMPATIBLE

    def summary_line(self) -> str:
        if self.status is not Compatibility.COMPATIBLE:
            return f"Measurement compatibility: {self.status.value}; " + "; ".join(
                self.reasons
            )
        shared = len(self.tasks.same) if self.tasks else 0
        candidate = (
            "candidate changed"
            + (f" ({', '.join(sorted(self.candidate_diff))})" if self.candidate_diff else "")
            if self.candidate_changed
            else "candidate unchanged"
        )
        line = (
            f"Measurement compatibility: compatible; {shared} shared task(s), "
            f"same graders; {candidate}"
        )
        if self.notes:
            line += "; note: " + "; ".join(self.notes)
        return line


def _list_ids(ids: Sequence[str], limit: int = 8) -> str:
    shown = ", ".join(ids[:limit])
    if len(ids) > limit:
        shown += f", and {len(ids) - limit} more"
    return shown


def check_compatibility(
    a: RunProvenance | None,
    b: RunProvenance | None,
) -> CompatibilityReport:
    """Decide whether two runs are comparable and describe what differs.

    Measurement differences (task content, task set, graders) make the
    runs *incompatible*; runner-setting and TraceLens-version differences
    are *notes*; candidate differences (adapter, ``DecisionSpec``) are
    reported separately and never affect the status. Missing provenance on
    either side yields *unknown*, never a silent match on task ids.
    """
    missing = [label for label, p in (("A", a), ("B", b)) if p is None]
    if a is None or b is None:
        return CompatibilityReport(
            status=Compatibility.UNKNOWN,
            reasons=[
                f"run {label} carries no provenance (artifact written before "
                "provenance was recorded, or by another producer); measurement "
                "compatibility cannot be established"
                for label in missing
            ],
        )

    reasons: list[str] = []
    notes: list[str] = []
    ma, mb = a.measurement, b.measurement

    tasks = TaskAlignment.between(ma.task_hashes, mb.task_hashes)
    if tasks.changed:
        reasons.append(
            f"{len(tasks.changed)} task(s) share an id but differ in content: "
            + _list_ids(tasks.changed)
        )
    if tasks.only_in_a:
        reasons.append(
            f"{len(tasks.only_in_a)} task(s) only in A: " + _list_ids(tasks.only_in_a)
        )
    if tasks.only_in_b:
        reasons.append(
            f"{len(tasks.only_in_b)} task(s) only in B: " + _list_ids(tasks.only_in_b)
        )
    if not (tasks.changed or tasks.only_in_a or tasks.only_in_b) and (
        ma.eval_set_hash != mb.eval_set_hash
    ):
        reasons.append(
            "eval-set content hashes differ "
            f"({short_hash(ma.eval_set_hash)} vs {short_hash(mb.eval_set_hash)}) "
            "and no per-task detail is available"
        )

    graders_a = sorted(g.describe() for g in ma.graders)
    graders_b = sorted(g.describe() for g in mb.graders)
    graders_changed = graders_a != graders_b
    if graders_changed:
        reasons.append(
            f"graders differ: A = [{', '.join(graders_a)}]; B = [{', '.join(graders_b)}]"
        )

    for name, va in ma.runner.model_dump().items():
        vb = getattr(mb.runner, name)
        if va != vb:
            notes.append(f"runner {name} differs ({va!r} vs {vb!r})")
    if a.tracelens_version != b.tracelens_version:
        notes.append(
            f"TraceLens version differs ({a.tracelens_version} vs {b.tracelens_version})"
        )

    ca, cb = a.candidate, b.candidate
    adapter_changed = ca.adapter != cb.adapter
    spec_changed = ca.decision_spec_fingerprint != cb.decision_spec_fingerprint
    candidate_diff: dict[str, list[Any]] = {}
    if ca.decision_spec is not None and cb.decision_spec is not None:
        candidate_diff = {
            key: [left, right]
            for key, (left, right) in sorted(ca.decision_spec.diff(cb.decision_spec).items())
        }
    elif spec_changed:
        notes.append("a DecisionSpec is declared on one side only; no candidate diff")

    return CompatibilityReport(
        status=Compatibility.INCOMPATIBLE if reasons else Compatibility.COMPATIBLE,
        reasons=reasons,
        notes=notes,
        tasks=tasks,
        graders_changed=graders_changed,
        adapter_changed=adapter_changed,
        candidate_changed=adapter_changed or spec_changed,
        candidate_diff=candidate_diff,
    )
