"""Tests for examples.graders.quality_grader."""

from __future__ import annotations

import json

import pytest

from examples.graders.quality_grader import QualityGrader
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript
from tracelens.llm.provider import InMemoryProvider


@pytest.mark.asyncio
async def test_quality_grader_zero_arg_offline_fallback() -> None:
    """QualityGrader can be instantiated with zero args and runs offline."""
    grader = QualityGrader()
    task = Task(
        task_id="t1",
        name="Test Task",
        input_data={"prompt": "Explain Python lists."},
    )
    transcript = Transcript(
        task_id="t1",
        final_output={"answer": "Lists are mutable ordered sequences in Python."},
    )

    outcome = await grader.grade(transcript, task)

    assert outcome.passed is True
    assert outcome.score >= 0.7
    assert outcome.grader_id == "quality"
    assert "completeness" in outcome.metrics
    assert "clarity" in outcome.metrics
    assert "accuracy" in outcome.metrics
    assert outcome.feedback != ""


@pytest.mark.asyncio
async def test_quality_grader_custom_provider() -> None:
    """QualityGrader accepts an explicit provider and parses its JSON response."""
    custom_canned = json.dumps(
        {
            "completeness": 4,
            "clarity": 5,
            "accuracy": 3,
            "feedback": "Output is incomplete and contains inaccuracies.",
        }
    )
    provider = InMemoryProvider(responses=[custom_canned])
    grader = QualityGrader(grader_id="custom_quality", provider=provider)

    task = Task(task_id="t2", name="Math problem", input_data={"prompt": "2+2"})
    transcript = Transcript(task_id="t2", final_output={"answer": 5})

    outcome = await grader.grade(transcript, task)

    assert outcome.passed is False
    assert outcome.score == pytest.approx(0.4)
    assert outcome.metrics["completeness"] == 0.4
    assert outcome.metrics["clarity"] == 0.5
    assert outcome.metrics["accuracy"] == 0.3
    assert "incomplete" in outcome.feedback
