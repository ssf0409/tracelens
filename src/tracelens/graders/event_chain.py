"""EventChainVerifier — a CodeGrader that checks for expected event sequences in transcripts.

Verifies that specific tool calls, outputs, or step types occurred
in the expected order during agent execution.
"""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from tracelens.core.grader import CodeGrader
from tracelens.core.task import Task
from tracelens.core.transcript import StepType, Transcript, TranscriptStep


class EventMatchType(StrEnum):
    """How to match a transcript step against an expectation."""

    TOOL_NAME = "tool_name"
    TOOL_NAME_AND_ARGS = "tool_name_and_args"
    CONTENT_REGEX = "content_regex"
    STEP_TYPE = "step_type"
    RESULT_REGEX = "result_regex"


class OrderingMode(StrEnum):
    """How to enforce ordering of matched events."""

    STRICT = "strict"        # All events must appear in exact order
    UNORDERED = "unordered"  # All events must appear, any order
    PARTIAL = "partial"      # Respects per-event `after` constraints (DAG)


class EventExpectation(BaseModel):
    """An expected event in the transcript."""

    event_id: str
    match_type: EventMatchType
    tool_name: str | None = None
    argument_patterns: dict[str, str] | None = None
    content_pattern: str | None = None
    step_type: StepType | None = None
    after: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_match_type_fields(self) -> "EventExpectation":
        mt = self.match_type
        if mt in (EventMatchType.TOOL_NAME, EventMatchType.TOOL_NAME_AND_ARGS):
            if not self.tool_name:
                raise ValueError(f"match_type='{mt}' requires 'tool_name'")
        if mt in (EventMatchType.CONTENT_REGEX, EventMatchType.RESULT_REGEX):
            if not self.content_pattern:
                raise ValueError(f"match_type='{mt}' requires 'content_pattern'")
        if mt == EventMatchType.STEP_TYPE:
            if self.step_type is None:
                raise ValueError("match_type='step_type' requires 'step_type'")
        return self


class EventChainConfig(BaseModel):
    """Configuration for EventChainVerifier."""

    expected_events: list[EventExpectation]
    ordering: OrderingMode = OrderingMode.STRICT
    require_all: bool = True
    score_per_event: bool = True


class EventChainVerifier(CodeGrader):
    """Verifies expected event sequences in transcripts.

    Scans transcript steps to match expected events, checks ordering
    constraints, and scores based on how many events were found.

    Example:
        config = EventChainConfig(
            expected_events=[
                EventExpectation(
                    event_id="search",
                    match_type=EventMatchType.TOOL_NAME,
                    tool_name="search",
                ),
                EventExpectation(
                    event_id="analyze",
                    match_type=EventMatchType.TOOL_NAME,
                    tool_name="analyze",
                    after=["search"],
                ),
            ],
            ordering=OrderingMode.PARTIAL,
        )
        verifier = EventChainVerifier("chain_check", config)
    """

    def __init__(self, grader_id: str, chain_config: EventChainConfig, **kwargs: Any) -> None:
        super().__init__(grader_id, **kwargs)
        self.chain_config = chain_config

    def compute_metrics(
        self,
        transcript: Transcript,
        task: Task,
    ) -> dict[str, float]:
        """Scan transcript and match against expected events.

        Uses greedy first-match: each step is matched against the first
        unmatched expectation it satisfies. Order expectations carefully
        when multiple expectations could match the same step.
        """
        expected = self.chain_config.expected_events
        found_ids: list[str] = []
        found_positions: dict[str, int] = {}

        for step_idx, step in enumerate(transcript.steps):
            for expectation in expected:
                if expectation.event_id in found_ids:
                    continue
                if self._step_matches(step, expectation):
                    found_ids.append(expectation.event_id)
                    found_positions[expectation.event_id] = step_idx
                    break

        missing_ids = [e.event_id for e in expected if e.event_id not in found_ids]
        total = len(expected)
        found_count = len(found_ids)

        ordering_correct = self._check_ordering(expected, found_ids, found_positions)

        return {
            "events_found": float(found_count),
            "events_missing": float(len(missing_ids)),
            "events_total": float(total),
            "events_found_ratio": found_count / total if total > 0 else 1.0,
            "ordering_correct": 1.0 if ordering_correct else 0.0,
        }

    def determine_pass(
        self,
        metrics: dict[str, float],
        task: Task,
    ) -> tuple[bool, float]:
        """Determine pass/fail from event matching metrics."""
        ratio = metrics["events_found_ratio"]
        ordering_ok = metrics["ordering_correct"] == 1.0

        if self.chain_config.require_all:
            passed = ratio == 1.0 and ordering_ok
        else:
            passed = ratio >= (self.config.pass_threshold if self.config else 0.5)

        score = ratio
        if not ordering_ok:
            score *= 0.5

        return passed, score

    def _step_matches(self, step: TranscriptStep, expectation: EventExpectation) -> bool:
        """Check if a transcript step matches an event expectation."""
        match expectation.match_type:
            case EventMatchType.TOOL_NAME:
                return (
                    step.tool_call is not None
                    and step.tool_call.tool_name == expectation.tool_name
                )
            case EventMatchType.TOOL_NAME_AND_ARGS:
                if step.tool_call is None or step.tool_call.tool_name != expectation.tool_name:
                    return False
                if expectation.argument_patterns:
                    return self._args_match(step.tool_call.arguments, expectation.argument_patterns)
                return True
            case EventMatchType.CONTENT_REGEX:
                if expectation.content_pattern is None:
                    return False
                content_str = str(step.content) if step.content is not None else ""
                return self._safe_search(expectation.content_pattern, content_str)
            case EventMatchType.STEP_TYPE:
                return step.step_type == expectation.step_type
            case EventMatchType.RESULT_REGEX:
                if expectation.content_pattern is None:
                    return False
                if step.tool_call is None or step.tool_call.result is None:
                    return False
                result_str = str(step.tool_call.result)
                return self._safe_search(expectation.content_pattern, result_str)
            case _:
                return False

    @staticmethod
    def _safe_search(pattern: str, text: str) -> bool:
        """Run regex search with error handling for invalid patterns."""
        try:
            return bool(re.search(pattern, text))
        except re.error:
            return False

    def _args_match(self, actual: dict[str, Any], patterns: dict[str, str]) -> bool:
        """Check if tool call arguments match patterns. Supports 're:' prefix for regex."""
        for key, pattern in patterns.items():
            if key not in actual:
                return False
            actual_val = str(actual[key])
            if pattern.startswith("re:"):
                if not self._safe_search(pattern[3:], actual_val):
                    return False
            elif actual_val != pattern:
                return False
        return True

    def _check_ordering(
        self,
        expected: list[EventExpectation],
        found_ids: list[str],
        found_positions: dict[str, int],
    ) -> bool:
        """Check if found events satisfy ordering constraints."""
        if self.chain_config.ordering == OrderingMode.UNORDERED:
            return True

        if self.chain_config.ordering == OrderingMode.STRICT:
            # All found events must appear in the same order as expected
            expected_order = [e.event_id for e in expected if e.event_id in found_ids]
            return found_ids == expected_order

        # PARTIAL: check per-event `after` DAG constraints
        for expectation in expected:
            if expectation.event_id not in found_positions:
                continue
            my_pos = found_positions[expectation.event_id]
            for dep_id in expectation.after:
                if dep_id not in found_positions:
                    return False
                if found_positions[dep_id] >= my_pos:
                    return False
        return True
