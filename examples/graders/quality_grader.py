"""Example LLMGrader for subjective quality evaluation.

Evaluates output quality using an LLM as judge. It is CLI-loadable with zero
arguments (e.g. ``tracelens run --graders examples.graders.quality_grader.QualityGrader``),
building its LLMProvider from environment variables:
- ``TRACELENS_JUDGE_PROVIDER``: ``"openai"``, ``"anthropic"``, or ``"in-memory"`` (default)
- ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` for live provider calls.
When no live provider is configured or when running offline/in CI, it falls back
to an ``InMemoryProvider`` returning valid rubric JSON so commands run cleanly without keys.
"""

from __future__ import annotations

import json
import os
from typing import Any

from tracelens import LLMGrader, LLMProvider
from tracelens.core.task import Task
from tracelens.core.transcript import Transcript
from tracelens.llm.provider import InMemoryProvider


def _build_default_provider(model: str) -> LLMProvider:
    """Build a provider based on environment variables or fall back to in-memory."""
    provider_type = os.getenv("TRACELENS_JUDGE_PROVIDER", "").strip().lower()

    if provider_type == "openai" or (not provider_type and os.getenv("OPENAI_API_KEY")):
        try:
            from openai import AsyncOpenAI

            class OpenAIChatProvider(LLMProvider):
                def __init__(self, model_name: str) -> None:
                    self.model_name = model_name
                    self._client = AsyncOpenAI()

                async def complete(self, prompt: str, **kwargs: Any) -> str:
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": "Return only valid JSON for TraceLens grading.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=kwargs.get("temperature", 0),
                    )
                    content = response.choices[0].message.content
                    if content is None:
                        raise RuntimeError("OpenAI response did not include message content")
                    return content

            return OpenAIChatProvider(model or "gpt-4o-mini")
        except ImportError:
            pass

    if provider_type == "anthropic" or (not provider_type and os.getenv("ANTHROPIC_API_KEY")):
        try:
            from anthropic import AsyncAnthropic

            class AnthropicMessagesProvider(LLMProvider):
                def __init__(self, model_name: str) -> None:
                    self.model_name = model_name
                    self._client = AsyncAnthropic()

                async def complete(self, prompt: str, **kwargs: Any) -> str:
                    response = await self._client.messages.create(
                        model=self.model_name,
                        max_tokens=kwargs.get("max_tokens", 512),
                        temperature=kwargs.get("temperature", 0),
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text_parts = [
                        block.text
                        for block in response.content
                        if getattr(block, "type", None) == "text"
                    ]
                    if not text_parts:
                        raise RuntimeError("Anthropic response did not include text content")
                    return "\n".join(text_parts)

            return AnthropicMessagesProvider(model or "claude-haiku-4-5-20251001")
        except ImportError:
            pass

    # Offline / dry-run fallback returning valid rubric evaluations with variance
    responses = [
        json.dumps(
            {
                "completeness": 9,
                "clarity": 9,
                "accuracy": 9,
                "feedback": "Offline judge fallback: high quality response.",
            }
        ),
        json.dumps(
            {
                "completeness": 5,
                "clarity": 6,
                "accuracy": 4,
                "feedback": "Offline judge fallback: mediocre response with inaccuracies.",
            }
        ),
    ]
    return InMemoryProvider(responses=responses)


class QualityGrader(LLMGrader):
    """Evaluates output quality using an LLM as judge.

    Rubric dimensions:
        - completeness: Does the output address all parts of the task?
        - clarity: Is the output clear and well-organized?
        - accuracy: Is the output factually correct?

    Each dimension is scored 1-10. The overall score is the average
    normalized to 0-1. Passing threshold is 0.7 (7/10 average).

    This grader is CLI-loadable without arguments:
        tracelens run --graders examples.graders.quality_grader.QualityGrader
    """

    def __init__(
        self,
        grader_id: str = "quality",
        *,
        model: str = "gpt-4o-mini",
        provider: LLMProvider | None = None,
        **kwargs: Any,
    ) -> None:
        if provider is None:
            provider = _build_default_provider(model)
        super().__init__(grader_id=grader_id, model=model, provider=provider, **kwargs)

    def build_grading_prompt(
        self,
        transcript: Transcript,
        task: Task,
    ) -> str:
        return f"""You are an expert evaluator. Score the following output on three dimensions.

## Task
Name: {task.name}
Description: {task.description or 'N/A'}
Input: {json.dumps(task.input_data)}

## Agent Output
{json.dumps(transcript.final_output, indent=2)}

## Rubric
Score each dimension from 1 to 10:
- **completeness**: Does the output address all parts of the task?
- **clarity**: Is the output clear and well-organized?
- **accuracy**: Is the output factually correct?

## Response Format
Return ONLY valid JSON:
{{
    "completeness": <1-10>,
    "clarity": <1-10>,
    "accuracy": <1-10>,
    "feedback": "<brief explanation>"
}}"""

    def parse_llm_response(
        self,
        response: str,
        task: Task,
    ) -> tuple[bool, float, dict[str, float], str]:
        data = json.loads(response)

        metrics = {
            "completeness": float(data["completeness"]) / 10.0,
            "clarity": float(data["clarity"]) / 10.0,
            "accuracy": float(data["accuracy"]) / 10.0,
        }

        score = sum(metrics.values()) / len(metrics)
        passed = score >= 0.7
        feedback = data.get("feedback", "")

        return passed, score, metrics, feedback
