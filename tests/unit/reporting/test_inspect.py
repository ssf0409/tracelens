"""Tests for the failure inspection view (issue #52)."""

from __future__ import annotations

import json

from tracelens.core.outcome import Outcome
from tracelens.core.task import Task, TaskExpectation
from tracelens.core.transcript import StepType, ToolCall, Transcript, TranscriptStep
from tracelens.core.trial import Trial, TrialBatch, TrialStatus
from tracelens.reporting.inspect import (
    FAILURE_KINDS,
    InspectionReport,
    TrialKind,
    build_inspection,
    classify,
    excerpt,
    render_html,
    render_text,
    select_trials,
    trial_view,
)


def _trial(
    task_id: str,
    run_index: int = 0,
    *,
    status: TrialStatus = TrialStatus.COMPLETED,
    outcomes: list[tuple[str, bool, float]] | None = None,
    feedback: str | None = None,
    grader_error: bool = False,
    transcript: Transcript | None = None,
    error_message: str | None = None,
) -> Trial:
    trial = Trial(task_id=task_id, run_index=run_index, status=status, error_message=error_message)
    trial.transcript = transcript
    for grader_id, passed, score in outcomes or []:
        trial.add_outcome(Outcome(
            trial_id=trial.trial_id, grader_id=grader_id, passed=passed, score=score,
            feedback=feedback, grader_error=grader_error,
            metrics={"exact": 1.0 if passed else 0.0},
        ))
    return trial


def _batch(*trials: Trial) -> TrialBatch:
    batch = TrialBatch()
    for trial in trials:
        batch.add_trial(trial)
    return batch


def _transcript(task_id: str, *, steps: int = 0, final_output: object = None) -> Transcript:
    transcript = Transcript(task_id=task_id, final_output=final_output)
    for i in range(steps):
        transcript.add_step(TranscriptStep(
            step_type=StepType.LLM_CALL, content=f"step {i} content", tokens_in=3, tokens_out=2,
        ))
    return transcript


PASSED = _trial("a", outcomes=[("g", True, 1.0)])
FAILED = _trial("b", outcomes=[("g", False, 0.0)], feedback="answer was wrong")
TIMEOUT = _trial("c", status=TrialStatus.TIMEOUT)
INFRA = _trial("d", status=TrialStatus.INFRA_ERROR, error_message="connection refused")
CRASHED = _trial("e", outcomes=[("g", False, 0.0)], grader_error=True, feedback="Sub-grader 'g' crashed")
PENDING = _trial("f", status=TrialStatus.PENDING)


class TestClassify:
    def test_one_kind_per_trial(self):
        assert classify(PASSED) is TrialKind.PASSED
        assert classify(FAILED) is TrialKind.AGENT_FAILURE
        assert classify(TIMEOUT) is TrialKind.AGENT_FAILURE
        assert classify(INFRA) is TrialKind.INFRA_ERROR
        assert classify(CRASHED) is TrialKind.GRADER_ERROR
        assert classify(PENDING) is TrialKind.NOT_RUN
        assert classify(_trial("g", status=TrialStatus.SKIPPED)) is TrialKind.NOT_RUN

    def test_harness_causes_win_over_agent_failure(self):
        infra_with_outcome = _trial(
            "x", status=TrialStatus.INFRA_ERROR, outcomes=[("g", False, 0.0)], grader_error=True
        )
        assert classify(infra_with_outcome) is TrialKind.INFRA_ERROR
        failed_and_crashed = _trial("y", outcomes=[("g", False, 0.0)], grader_error=True)
        assert classify(failed_and_crashed) is TrialKind.GRADER_ERROR


class TestSelectTrials:
    def test_kinds_task_ids_and_order(self):
        batch = _batch(PENDING, CRASHED, INFRA, TIMEOUT, FAILED, PASSED)
        assert [t.task_id for t in select_trials(batch)] == ["a", "b", "c", "d", "e", "f"]
        assert [t.task_id for t in select_trials(batch, kinds=FAILURE_KINDS)] == ["b", "c", "d", "e"]
        assert [t.task_id for t in select_trials(batch, kinds=[TrialKind.INFRA_ERROR])] == ["d"]
        assert [t.task_id for t in select_trials(batch, kinds=FAILURE_KINDS, task_ids=["b", "z"])] == ["b"]

    def test_grader_filter_attributes_only_failures_and_crashes(self):
        mixed = Trial(task_id="m", status=TrialStatus.COMPLETED)
        mixed.add_outcome(Outcome(trial_id=mixed.trial_id, grader_id="g1", passed=True, score=1.0))
        mixed.add_outcome(Outcome(trial_id=mixed.trial_id, grader_id="g2", passed=False, score=0.2))
        batch = _batch(mixed, CRASHED, PASSED)
        assert [t.task_id for t in select_trials(batch, grader_ids=["g1"])] == []
        assert [t.task_id for t in select_trials(batch, grader_ids=["g2"])] == ["m"]
        assert [t.task_id for t in select_trials(batch, grader_ids=["g"])] == ["e"]


class TestTrialView:
    def test_expected_states_are_explicit(self):
        task = Task(task_id="b", name="named", input_data={"q": "2+2"},
                    expectation=TaskExpectation(expected_output="4"))
        bare = Task(task_id="b", name="bare", input_data={})
        assert trial_view(FAILED).expected == "not supplied (pass --eval-set to show it)"
        assert trial_view(FAILED, eval_set_supplied=True).expected == (
            "missing (task not in the supplied eval set)"
        )
        assert trial_view(FAILED, task=bare, eval_set_supplied=True).expected == (
            "missing (the task declares no expected output)"
        )
        view = trial_view(FAILED, task=task, eval_set_supplied=True)
        assert view.expected == "4" and view.task_name == "named"
        assert view.task_input == '{"q": "2+2"}'

    def test_missing_and_zero_values_render_explicitly(self):
        view = trial_view(FAILED)
        assert view.actual == "missing" and view.transcript is None
        assert view.outcomes[0].describe() == (
            "g FAIL score=0.00 metrics=exact=0 feedback: answer was wrong"
        )
        crashed = trial_view(CRASHED).outcomes[0]
        assert crashed.grader_error and crashed.describe().startswith("g CRASHED score=0.00")
        timeout = trial_view(TIMEOUT)
        assert timeout.kind is TrialKind.AGENT_FAILURE and timeout.status == "timeout"
        assert timeout.outcomes == []
        assert trial_view(INFRA).error_message == "connection refused"
        no_feedback = trial_view(_trial("n", outcomes=[("g", False, 0.0)]))
        assert no_feedback.outcomes[0].feedback == "missing"

    def test_multi_grader_and_transcript_summary(self):
        trial = _trial("t", outcomes=[("g1", True, 1.0), ("g2", False, 0.4)],
                       transcript=_transcript("t", steps=3, final_output={"answer": 3}))
        trial.transcript.add_step(TranscriptStep(  # type: ignore[union-attr]
            step_type=StepType.TOOL_CALL, error="boom",
            tool_call=ToolCall(tool_name="search", arguments={"q": "x"}, result=None, error="boom"),
        ))
        view = trial_view(trial)
        assert [o.grader_id for o in view.outcomes] == ["g1", "g2"]
        assert view.actual == '{"answer": 3}'
        assert view.transcript is not None
        assert view.transcript.headline().startswith("4 step(s) (4 shown), 15 tokens, 3 llm call(s), 1 tool call(s)")
        assert view.transcript.steps[3].summary == 'tool search({"q": "x"}) -> missing [tool error: boom]'
        assert view.transcript.errors == ["boom"]
        assert view.transcript.steps[3].describe().endswith("ERROR: boom")
        assert trial_view(_trial("e", transcript=_transcript("e"))).transcript.headline().startswith("0 step(s) (0 shown)")  # type: ignore[union-attr]

    def test_bounds_count_what_they_omit(self):
        long_output = "x" * 1000
        trial = _trial("t", outcomes=[("g", False, 0.0)], feedback="f" * 50,
                       transcript=_transcript("t", steps=30, final_output=long_output))
        view = trial_view(trial, max_steps=5, max_chars=100)
        assert view.actual == "x" * 100 + "… (900 more characters)"
        assert view.transcript is not None
        assert (view.transcript.steps_shown, view.transcript.steps_omitted) == (5, 25)
        assert "5 shown, 25 omitted" in view.transcript.headline()
        assert view.outcomes[0].feedback == "f" * 50 and not view.outcomes[0].feedback_truncated
        full = trial_view(trial, max_steps=None, max_chars=None)
        assert full.actual == long_output and full.transcript.steps_omitted == 0  # type: ignore[union-attr]
        assert excerpt(None, 10) == ("missing", False)
        assert excerpt({"b": 1, "a": 2}, None) == ('{"a": 2, "b": 1}', False)


class TestBuildAndRender:
    def _report(self, **kwargs: object) -> InspectionReport:
        batch = _batch(PASSED, FAILED, TIMEOUT, INFRA, CRASHED, PENDING)
        return build_inspection(batch, source="trials.json", **kwargs)  # type: ignore[arg-type]

    def test_totals_selection_and_limit(self):
        report = self._report()
        assert report.totals == {
            TrialKind.PASSED: 1, TrialKind.AGENT_FAILURE: 2, TrialKind.INFRA_ERROR: 1,
            TrialKind.GRADER_ERROR: 1, TrialKind.NOT_RUN: 1,
        }
        assert report.selected == 4 and report.shown == 4
        assert [t.task_id for t in report.trials] == ["b", "c", "d", "e"]
        limited = self._report(limit=2)
        assert (limited.selected, limited.shown) == (4, 2)
        everything = self._report(kinds=None)
        assert everything.selected == 6 and everything.selection == "all trials"
        full = self._report(full=True)
        assert full.max_steps is None and full.max_chars is None

    def test_text_is_plain_and_explicit(self):
        text = render_text(self._report(limit=2))
        assert text.startswith("Inspected trials.json: 6 trial(s), run missing, TraceLens ")
        assert "passed 1, agent failure 2, infra error 1, grader error 1, not run 1" in text
        assert "Selected 4 trial(s) (kinds: agent failure, infra error, grader error); showing the first 2" in text
        assert "expected outputs: not supplied (pass --eval-set to show them)" in text
        assert "[1] b run 0  agent failure  status=completed  attempts=1" in text
        assert "why:      the agent ran and a grader failed it (a timeout counts)" in text
        assert "grader:   g FAIL score=0.00 metrics=exact=0 feedback: answer was wrong" in text
        assert "transcript: missing" in text
        assert "graders:  none (no outcome was recorded)" in text  # the timeout
        assert text.rstrip().endswith("Pass --full to include everything.")

    def test_text_with_eval_set_and_kinds(self):
        tasks = [Task(task_id="d", name="infra task", input_data={"n": 1})]
        text = render_text(self._report(kinds=[TrialKind.INFRA_ERROR], tasks=tasks))
        assert "task:     infra task" in text and "input:    {\"n\": 1}" in text
        assert "error:    connection refused" in text
        assert "expected: missing (the task declares no expected output)" in text
        assert "why:      infrastructure failed before the agent could be judged" in text

    def test_html_is_escaped_offline_and_bounded(self):
        hostile = _trial("<script>alert(1)</script>", outcomes=[("g", False, 0.0)],
                         feedback="<b>bold</b>",
                         transcript=_transcript("h", final_output="SECRET-" + "s" * 600))
        batch = _batch(hostile)
        html = render_html(build_inspection(batch, source="t.json"))
        assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
        assert "<b>bold</b>" not in html and "&lt;b&gt;bold&lt;/b&gt;" in html
        assert "src=" not in html and "href=" not in html  # nothing fetched from anywhere
        assert 'name="viewport"' in html
        assert "SECRET-" + "s" * 600 not in html and "more characters" in html
        assert "agent failure 1" in html and "<details class=\"trial\" open>" in html
        unbounded = render_html(build_inspection(batch, source="t.json", full=True))
        assert "SECRET-" + "s" * 600 in unbounded and "Unbounded output (--full)" in unbounded
        empty = render_html(build_inspection(_batch(PASSED), source="t.json"))
        assert "No trial matches the selection." in empty

    def test_json_round_trip(self):
        report = self._report()
        data = json.loads(report.model_dump_json())
        assert InspectionReport.model_validate(data) == report
        assert data["totals"]["agent_failure"] == 2 and data["trials"][0]["kind"] == "agent_failure"
