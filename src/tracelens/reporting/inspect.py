"""Explain failed trials from a saved trials artifact (issue #52).

``tracelens inspect trials.json --failures`` turns the raw JSON written by
``tracelens run --save-trials`` into a reading order: which trials failed and
*why* (agent failure, infrastructure error, or grader crash, never
conflated), what was expected and what came back, what each grader said, and
what the transcript did. Every field that is absent is shown as ``missing``
rather than left blank, and long content is bounded with an explicit count of
what was left out, so the view never misreports what it omitted.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from enum import StrEnum
from html import escape
from typing import Any

from pydantic import BaseModel, Field

from tracelens._version import __version__
from tracelens.core.provenance import task_content_hash
from tracelens.core.task import Task
from tracelens.core.transcript import StepType, Transcript, TranscriptStep
from tracelens.core.trial import Trial, TrialBatch, TrialStatus

MISSING = "missing"
DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_CHARS = 400


class TaskContextError(ValueError):
    """Raised when task context resolution or validation fails."""

    def __init__(
        self,
        message: str,
        *,
        reason: str | None = None,
        task_id: str | None = None,
        task_ids: list[str] | None = None,
        recorded_hash: str | None = None,
        computed_hash: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.task_id = task_id or (task_ids[0] if task_ids else None)
        self.task_ids = task_ids or ([task_id] if task_id else [])
        self.recorded_hash = recorded_hash
        self.computed_hash = computed_hash


class TaskDuplicateIdError(TaskContextError):
    """Raised when an eval set contains duplicate task IDs."""

    def __init__(
        self,
        task_ids: list[str],
        *,
        message: str | None = None,
    ) -> None:
        seen = set()
        dupes = []
        for i in task_ids:
            if i in seen:
                dupes.append(i)
            seen.add(i)

        if message is None:
            formatted = ", ".join(repr(i) for i in sorted(set(dupes)))
            message = f"duplicate task ID(s) in eval set: {formatted}; each task ID must be unique"
        super().__init__(
            message,
            task_ids=dupes,
            reason="duplicate_task_id",
        )


class TaskContentMismatchError(TaskContextError):
    """Raised when a task in the eval set does not match the recorded hash."""

    def __init__(
        self,
        task_id: str,
        *,
        recorded_hash: str,
        computed_hash: str,
        message: str | None = None,
    ) -> None:
        if message is None:
            message = (
                f"task {task_id!r} content does not match the task recorded in the trials "
                f"(recorded hash {recorded_hash[:12]}, eval set hash {computed_hash[:12]}). "
                "Supply the original eval set used for the run, or run without --eval-set "
                "to inspect the recorded execution evidence alone."
            )
        super().__init__(
            message,
            reason="content_mismatch",
            task_id=task_id,
            recorded_hash=recorded_hash,
            computed_hash=computed_hash,
        )



class TaskContextStatus(StrEnum):
    """Verification status of attached task context."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    NONE = "none"


class TrialKind(StrEnum):
    """What a trial says, for filtering: agent, harness, or nothing."""

    PASSED = "passed"
    AGENT_FAILURE = "agent_failure"
    INFRA_ERROR = "infra_error"
    GRADER_ERROR = "grader_error"
    NOT_RUN = "not_run"


FAILURE_KINDS = (TrialKind.AGENT_FAILURE, TrialKind.INFRA_ERROR, TrialKind.GRADER_ERROR)
KIND_FLAGS: dict[str, TrialKind] = {
    "agent": TrialKind.AGENT_FAILURE,
    "infra": TrialKind.INFRA_ERROR,
    "grader": TrialKind.GRADER_ERROR,
    "not-run": TrialKind.NOT_RUN,
    "passed": TrialKind.PASSED,
}
KIND_LABELS: dict[TrialKind, str] = {
    TrialKind.PASSED: "passed",
    TrialKind.AGENT_FAILURE: "agent failure",
    TrialKind.INFRA_ERROR: "infra error",
    TrialKind.GRADER_ERROR: "grader error",
    TrialKind.NOT_RUN: "not run",
}
KIND_MEANING: dict[TrialKind, str] = {
    TrialKind.PASSED: "every grader passed the trial",
    TrialKind.AGENT_FAILURE: "the agent ran and a grader failed it (a timeout counts)",
    TrialKind.INFRA_ERROR: "infrastructure failed before the agent could be judged",
    TrialKind.GRADER_ERROR: "a grader crashed; this says nothing about the agent",
    TrialKind.NOT_RUN: "the trial never ran (pending, running, or skipped)",
}


def classify(trial: Trial) -> TrialKind:
    """One kind per trial, harness causes first so they are never conflated."""
    if trial.status is TrialStatus.INFRA_ERROR:
        return TrialKind.INFRA_ERROR
    if trial.has_grader_error:
        return TrialKind.GRADER_ERROR
    if not trial.is_gradable:
        return TrialKind.NOT_RUN
    return TrialKind.PASSED if trial.passed else TrialKind.AGENT_FAILURE


# --- bounded rendering of arbitrary values -------------------------------------


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)


def excerpt(value: Any, max_chars: int | None) -> tuple[str, bool]:
    """A bounded string for ``value`` and whether it was cut.

    ``None`` renders as ``missing``. When cut, the string ends with an explicit
    ``… (N more characters)`` marker so nothing is omitted silently.
    """
    if value is None:
        return MISSING, False
    text = _text(value)
    if max_chars is None or len(text) <= max_chars:
        return text, False
    return f"{text[:max_chars]}… ({len(text) - max_chars} more characters)", True


class OutcomeView(BaseModel):
    grader_id: str
    passed: bool
    score: float
    grader_error: bool = False
    grade_level: str | None = None
    feedback: str
    feedback_truncated: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)

    def describe(self) -> str:
        verdict = "CRASHED" if self.grader_error else ("PASS" if self.passed else "FAIL")
        text = f"{self.grader_id} {verdict} score={self.score:.2f}"
        if self.metrics:
            text += " metrics=" + ", ".join(f"{k}={v:g}" for k, v in sorted(self.metrics.items()))
        return text + f" feedback: {self.feedback}"


class StepView(BaseModel):
    index: int
    step_type: str
    summary: str
    truncated: bool = False
    error: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    def describe(self) -> str:
        text = f"{self.index}. {self.step_type}"
        if self.tokens_in is not None or self.tokens_out is not None:
            text += f"  tokens {self.tokens_in if self.tokens_in is not None else '?'}/"
            text += f"{self.tokens_out if self.tokens_out is not None else '?'}"
        text += f"  {self.summary}"
        if self.error:
            text += f"  ERROR: {self.error}"
        return text


class TranscriptView(BaseModel):
    steps_total: int
    steps_shown: int
    steps_omitted: int
    steps: list[StepView] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    final_output: str
    final_output_truncated: bool = False
    duration_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    agent_name: str | None = None

    def headline(self) -> str:
        text = (
            f"{self.steps_total} step(s) ({self.steps_shown} shown"
            + (f", {self.steps_omitted} omitted" if self.steps_omitted else "")
            + f"), {self.input_tokens + self.output_tokens} tokens, "
            f"{self.llm_calls} llm call(s), {self.tool_calls} tool call(s)"
        )
        if self.duration_ms is not None:
            text += f", {self.duration_ms:.0f} ms"
        return text


class TrialView(BaseModel):
    task_id: str
    task_name: str | None = None
    run_index: int
    trial_id: str
    status: str
    kind: TrialKind
    attempts: int = 1
    duration_ms: float | None = None
    error_message: str | None = None
    task_input: str | None = None
    expected: str
    actual: str
    outcomes: list[OutcomeView] = Field(default_factory=list)
    transcript: TranscriptView | None = None
    task_context_status: TaskContextStatus = TaskContextStatus.NONE

    def headline(self) -> str:
        text = (
            f"{self.task_id} run {self.run_index}  {KIND_LABELS[self.kind]}  "
            f"status={self.status}  attempts={self.attempts}"
        )
        if self.duration_ms is not None:
            text += f"  duration={self.duration_ms:.0f} ms"
        return text


class InspectionReport(BaseModel):
    """Everything the text and HTML views render; ``--json`` writes it as is."""

    source: str
    run_id: str | None = None
    tracelens_version: str = __version__
    total_trials: int
    totals: dict[TrialKind, int] = Field(default_factory=dict)
    selection: str
    selected: int
    shown: int
    eval_set_supplied: bool = False
    task_context_status: TaskContextStatus = TaskContextStatus.NONE
    max_steps: int | None = DEFAULT_MAX_STEPS
    max_chars: int | None = DEFAULT_MAX_CHARS
    full: bool = False
    trials: list[TrialView] = Field(default_factory=list)


# --- building the views ----------------------------------------------------------


def _step_summary(step: TranscriptStep, max_chars: int | None) -> tuple[str, bool]:
    if step.tool_call is not None:
        call = step.tool_call
        args, cut_args = excerpt(call.arguments, max_chars)
        result, cut_result = excerpt(call.result, max_chars)
        text = f"tool {call.tool_name}({args}) -> {result}"
        if call.error:
            text += f" [tool error: {call.error}]"
        return text, cut_args or cut_result
    content, cut = excerpt(step.content, max_chars)
    prefix = f"model={step.model} " if step.model else ""
    return f"{prefix}content: {content}", cut


def _transcript_view(
    transcript: Transcript, max_steps: int | None, max_chars: int | None
) -> TranscriptView:
    steps = transcript.steps
    shown = steps if max_steps is None else steps[:max_steps]
    views = []
    for index, step in enumerate(shown, start=1):
        summary, cut = _step_summary(step, max_chars)
        views.append(StepView(
            index=index,
            step_type=step.step_type.value,
            summary=summary,
            truncated=cut,
            error=step.error,
            tokens_in=step.tokens_in,
            tokens_out=step.tokens_out,
        ))
    final_output, final_cut = excerpt(transcript.final_output, max_chars)
    return TranscriptView(
        steps_total=len(steps),
        steps_shown=len(shown),
        steps_omitted=len(steps) - len(shown),
        steps=views,
        errors=[excerpt(e, max_chars)[0] for e in transcript.errors],
        final_output=final_output,
        final_output_truncated=final_cut,
        duration_ms=transcript.duration_ms,
        input_tokens=transcript.input_tokens,
        output_tokens=transcript.output_tokens,
        llm_calls=transcript.llm_calls_count,
        tool_calls=sum(1 for s in steps if s.step_type is StepType.TOOL_CALL),
        agent_name=transcript.agent_name,
    )


def trial_view(
    trial: Trial,
    *,
    task: Task | None = None,
    eval_set_supplied: bool = False,
    task_context_status: TaskContextStatus = TaskContextStatus.NONE,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    max_chars: int | None = DEFAULT_MAX_CHARS,
) -> TrialView:
    """The bounded, explicit view of one trial."""
    if task is not None and task.expectation is not None and (
        task.expectation.expected_output is not None
    ):
        expected, _ = excerpt(task.expectation.expected_output, max_chars)
    elif task is not None:
        expected = "missing (the task declares no expected output)"
    elif eval_set_supplied:
        expected = "missing (task not in the supplied eval set)"
    else:
        expected = "not supplied (pass --eval-set to show it)"
    transcript = trial.transcript
    actual, _ = excerpt(transcript.final_output if transcript is not None else None, max_chars)
    outcomes = []
    for outcome in trial.outcomes:
        feedback, cut = excerpt(outcome.feedback, max_chars)
        outcomes.append(OutcomeView(
            grader_id=outcome.grader_id,
            passed=outcome.passed,
            score=outcome.score,
            grader_error=outcome.grader_error,
            grade_level=outcome.grade_level.value if outcome.grade_level else None,
            feedback=feedback,
            feedback_truncated=cut,
            metrics=dict(outcome.metrics),
        ))
    return TrialView(
        task_id=trial.task_id,
        task_name=task.name if task is not None else None,
        run_index=trial.run_index,
        trial_id=trial.trial_id,
        status=trial.status.value,
        kind=classify(trial),
        attempts=trial.attempts,
        duration_ms=trial.duration_ms,
        error_message=excerpt(trial.error_message, max_chars)[0] if trial.error_message else None,
        task_input=excerpt(task.input_data, max_chars)[0] if task is not None else None,
        expected=expected,
        actual=actual,
        outcomes=outcomes,
        transcript=(
            _transcript_view(transcript, max_steps, max_chars) if transcript is not None else None
        ),
        task_context_status=task_context_status,
    )


def select_trials(
    batch: TrialBatch,
    *,
    kinds: Iterable[TrialKind] | None = None,
    task_ids: Sequence[str] | None = None,
    grader_ids: Sequence[str] | None = None,
) -> list[Trial]:
    """Trials in canonical order (task id, run index) matching every filter.

    ``grader_ids`` keeps trials that have an outcome from one of those
    graders which failed or crashed; a passing outcome from that grader does
    not qualify, and a trial the grader never saw is not attributed to it.
    """
    wanted = set(kinds) if kinds is not None else None
    ids = set(task_ids) if task_ids else None
    graders = set(grader_ids) if grader_ids else None
    selected = []
    for trial in sorted(batch.trials, key=lambda t: (t.task_id, t.run_index)):
        if wanted is not None and classify(trial) not in wanted:
            continue
        if ids is not None and trial.task_id not in ids:
            continue
        if graders is not None and not any(
            o.grader_id in graders and (o.grader_error or not o.passed) for o in trial.outcomes
        ):
            continue
        selected.append(trial)
    return selected


def validate_task_context(
    by_id: dict[str, Task],
    shown: Sequence[Trial],
    provenance_hashes: dict[str, str],
) -> dict[str, TaskContextStatus]:
    """Validate task set for hash mismatches and resolve per-task context statuses.

    Returns:
        Mapping from task_id to TaskContextStatus.
    """
    has_provenance_hashes = bool(provenance_hashes)
    attached_task_ids = {trial.task_id for trial in shown if trial.task_id in by_id}

    per_task_status: dict[str, TaskContextStatus] = {}

    for tid in attached_task_ids:
        task_obj = by_id[tid]
        if not has_provenance_hashes or tid not in provenance_hashes:
            per_task_status[tid] = TaskContextStatus.UNVERIFIED
            continue

        recorded_hash = provenance_hashes[tid]
        computed_hash = task_content_hash(task_obj)
        if recorded_hash != computed_hash:
            raise TaskContentMismatchError(
                tid, recorded_hash=recorded_hash, computed_hash=computed_hash
            )
        per_task_status[tid] = TaskContextStatus.VERIFIED

    return per_task_status


def build_inspection(
    batch: TrialBatch,
    *,
    source: str,
    kinds: Iterable[TrialKind] | None = FAILURE_KINDS,
    task_ids: Sequence[str] | None = None,
    grader_ids: Sequence[str] | None = None,
    tasks: Sequence[Task] | None = None,
    max_steps: int | None = DEFAULT_MAX_STEPS,
    max_chars: int | None = DEFAULT_MAX_CHARS,
    full: bool = False,
    limit: int | None = None,
) -> InspectionReport:
    """Select and render trials into an :class:`InspectionReport`.

    ``kinds=None`` selects every trial; the default selects the three failure
    kinds. ``full`` disables all bounds (the output then embeds complete
    transcripts, which may be sensitive).
    """
    if full:
        max_steps = None
        max_chars = None
    totals = {kind: 0 for kind in TrialKind}
    for trial in batch.trials:
        totals[classify(trial)] += 1
    selected = select_trials(batch, kinds=kinds, task_ids=task_ids, grader_ids=grader_ids)
    shown = selected if limit is None else selected[:limit]

    if tasks is None:
        by_id: dict[str, Task] = {}
        per_task_status: dict[str, TaskContextStatus] = {}
        report_context_status = TaskContextStatus.NONE
    else:
        by_id = {task.task_id: task for task in tasks}
        if len(tasks) != len(by_id):
            raise TaskDuplicateIdError(task_ids=[t.task_id for t in tasks])

        task_hashes = (
            batch.provenance.measurement.task_hashes
            if (batch.provenance is not None and batch.provenance.measurement is not None)
            else {}
        )
        per_task_status = validate_task_context(by_id, shown, task_hashes)
        has_unverified = TaskContextStatus.UNVERIFIED in per_task_status.values()
        report_context_status = (
            TaskContextStatus.UNVERIFIED if has_unverified else TaskContextStatus.VERIFIED
        )

    parts = []
    if kinds is None:
        parts.append("all trials")
    else:
        parts.append("kinds: " + ", ".join(KIND_LABELS[k] for k in kinds))
    if task_ids:
        parts.append("task ids: " + ", ".join(task_ids))
    if grader_ids:
        parts.append("failed by grader: " + ", ".join(grader_ids))
    return InspectionReport(
        source=source,
        run_id=batch.provenance.run_id if batch.provenance is not None else None,
        total_trials=batch.total_count,
        totals=totals,
        selection="; ".join(parts),
        selected=len(selected),
        shown=len(shown),
        eval_set_supplied=tasks is not None,
        task_context_status=report_context_status,
        max_steps=max_steps,
        max_chars=max_chars,
        full=full,
        trials=[
            trial_view(
                trial,
                task=by_id.get(trial.task_id),
                eval_set_supplied=tasks is not None,
                task_context_status=per_task_status.get(trial.task_id, TaskContextStatus.NONE),
                max_steps=max_steps,
                max_chars=max_chars,
            )
            for trial in shown
        ],
    )


# --- text rendering ---------------------------------------------------------------


def _bounds_note(report: InspectionReport) -> str:
    if report.full:
        return "Unbounded output (--full): complete transcripts are included."
    return (
        f"Output is bounded to {report.max_chars} characters per field and "
        f"{report.max_steps} steps per transcript; omitted content is counted, never "
        "hidden. Pass --full to include everything."
    )


def render_text(report: InspectionReport) -> str:
    """Plain, non-TTY text: what failed, why, and what to read next."""
    run = report.run_id or MISSING
    lines = [
        f"Inspected {report.source}: {report.total_trials} trial(s), run {run}, "
        f"TraceLens {report.tracelens_version}",
        "  " + ", ".join(
            f"{KIND_LABELS[kind]} {report.totals.get(kind, 0)}" for kind in TrialKind
        ),
        f"Selected {report.selected} trial(s) ({report.selection})"
        + (f"; showing the first {report.shown}" if report.shown < report.selected else ""),
    ]
    if not report.eval_set_supplied:
        lines.append("  expected outputs: not supplied (pass --eval-set to show them)")
    elif report.task_context_status == TaskContextStatus.UNVERIFIED:
        lines.append("  expected outputs: unverified (trials file carries no task hashes)")
    for number, trial in enumerate(report.trials, start=1):
        lines.append("")
        lines.append(f"[{number}] {trial.headline()}")
        lines.append(f"    why:      {KIND_MEANING[trial.kind]}")
        task_label = trial.task_name
        if trial.task_context_status == TaskContextStatus.UNVERIFIED and task_label:
            task_label += " (unverified)"
        if task_label:
            lines.append(f"    task:     {task_label}")
        if trial.task_input is not None:
            lines.append(f"    input:    {trial.task_input}")
        if trial.error_message:
            lines.append(f"    error:    {trial.error_message}")
        lines.append(f"    expected: {trial.expected}")
        lines.append(f"    actual:   {trial.actual}")
        if trial.outcomes:
            for outcome in trial.outcomes:
                lines.append(f"    grader:   {outcome.describe()}")
        else:
            lines.append("    graders:  none (no outcome was recorded)")
        if trial.transcript is None:
            lines.append("    transcript: missing")
        else:
            lines.append(f"    transcript: {trial.transcript.headline()}")
            for step in trial.transcript.steps:
                lines.append(f"      {step.describe()}")
            for error in trial.transcript.errors:
                lines.append(f"      transcript error: {error}")
    lines.append("")
    lines.append(_bounds_note(report))
    return "\n".join(lines)


# --- HTML rendering ---------------------------------------------------------------

_KIND_COLORS: dict[TrialKind, str] = {
    TrialKind.PASSED: "#22c55e",
    TrialKind.AGENT_FAILURE: "#ef4444",
    TrialKind.INFRA_ERROR: "#f97316",
    TrialKind.GRADER_ERROR: "#a855f7",
    TrialKind.NOT_RUN: "#6b7280",
}


def _row(label: str, value: str) -> str:
    return f"<tr><th>{escape(label)}</th><td>{escape(value)}</td></tr>"


def _trial_html(number: int, trial: TrialView) -> str:
    color = _KIND_COLORS[trial.kind]
    rows = [_row("why", KIND_MEANING[trial.kind])]
    if trial.task_name:
        rows.append(_row(
            "task",
            f"{trial.task_name} (unverified)"
            if trial.task_context_status == TaskContextStatus.UNVERIFIED
            else trial.task_name,
        ))

    if trial.task_input is not None:
        rows.append(_row("input", trial.task_input))
    if trial.error_message:
        rows.append(_row("error", trial.error_message))
    rows.append(_row("expected", trial.expected))
    rows.append(_row("actual", trial.actual))
    rows.append(_row("trial id", trial.trial_id))
    graders = "".join(
        f"<li><span class=\"badge\" style=\"background:{'#a855f7' if o.grader_error else ('#22c55e' if o.passed else '#ef4444')}\">"
        f"{'CRASHED' if o.grader_error else ('PASS' if o.passed else 'FAIL')}</span> "
        f"{escape(o.describe())}</li>"
        for o in trial.outcomes
    ) or "<li>none (no outcome was recorded)</li>"
    if trial.transcript is None:
        transcript = "<p class=\"muted\">transcript: missing</p>"
    else:
        steps = "".join(
            f"<li><code>{escape(step.describe())}</code></li>" for step in trial.transcript.steps
        ) or "<li class=\"muted\">no steps recorded</li>"
        errors = "".join(
            f"<li class=\"error\">{escape(error)}</li>" for error in trial.transcript.errors
        )
        transcript = (
            f"<details><summary>transcript: {escape(trial.transcript.headline())}</summary>"
            f"<ol class=\"steps\">{steps}</ol>"
            + (f"<ul class=\"errors\">{errors}</ul>" if errors else "")
            + "</details>"
        )
    return (
        f"<details class=\"trial\" open><summary><span class=\"badge\" style=\"background:{color}\">"
        f"{escape(KIND_LABELS[trial.kind])}</span> [{number}] {escape(trial.headline())}</summary>"
        f"<table>{''.join(rows)}</table><h4>Graders</h4><ul class=\"graders\">{graders}</ul>"
        f"{transcript}</details>"
    )


def render_html(report: InspectionReport) -> str:
    """A self-contained, offline drilldown; every value is escaped."""
    counts = "".join(
        f"<span class=\"badge\" style=\"background:{_KIND_COLORS[kind]}\">"
        f"{escape(KIND_LABELS[kind])} {report.totals.get(kind, 0)}</span>"
        for kind in TrialKind
    )
    trials = "".join(_trial_html(n, t) for n, t in enumerate(report.trials, start=1)) or (
        "<p class=\"muted\">No trial matches the selection.</p>"
    )
    showing = (
        f"; showing the first {report.shown}" if report.shown < report.selected else ""
    )
    if not report.eval_set_supplied:
        expected_note = "<p class=\"muted\">Expected outputs: not supplied (pass --eval-set to show them).</p>"
    elif report.task_context_status == TaskContextStatus.UNVERIFIED:
        expected_note = "<p class=\"muted\">Expected outputs: unverified (trials file carries no task hashes).</p>"
    else:
        expected_note = ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraceLens Inspection</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 16px;
         background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 1.4em; margin: 0 0 4px; }}
  h4 {{ margin: 12px 0 4px; font-size: 0.95em; }}
  .muted {{ color: #64748b; font-size: 0.9em; }}
  .badge {{ color: #fff; padding: 2px 8px; border-radius: 10px; font-size: 0.75em;
            margin-right: 6px; white-space: nowrap; }}
  .counts {{ margin: 10px 0 16px; display: flex; flex-wrap: wrap; gap: 6px; }}
  details.trial {{ background: #fff; border-radius: 8px; padding: 10px 14px;
                   box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 12px; }}
  details.trial > summary {{ cursor: pointer; font-weight: 600; }}
  p, summary, td, li, code {{ overflow-wrap: anywhere; }}
  table {{ width: 100%; table-layout: fixed; border-collapse: collapse; font-size: 0.9em;
           margin-top: 8px; }}
  th {{ text-align: left; padding: 4px 8px; color: #475569; width: 7em; vertical-align: top; }}
  td {{ padding: 4px 8px; white-space: pre-wrap; }}
  ul, ol {{ padding-left: 1.4em; margin: 4px 0; }}
  li {{ margin: 2px 0; }}
  code {{ white-space: pre-wrap; font-size: 0.85em; }}
  .error {{ color: #b91c1c; }}
  @media (max-width: 600px) {{ body {{ padding: 8px; }} th {{ width: 6.5em; }} }}
</style>
</head>
<body>
<h1>TraceLens Inspection</h1>
<p class="muted">{escape(report.source)} &middot; {report.total_trials} trial(s) &middot; run {escape(report.run_id or MISSING)} &middot; TraceLens v{escape(report.tracelens_version)}</p>
<div class="counts">{counts}</div>
<p>Selected {report.selected} trial(s) ({escape(report.selection)}){escape(showing)}.</p>
{expected_note}
{trials}
<p class="muted">{escape(_bounds_note(report))}</p>
</body>
</html>"""
