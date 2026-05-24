"""OpenAI and Anthropic provider examples for LLM-as-judge grading.

This example is safe to run without API keys. By default it uses recorded
responses so you can inspect the provider shape and the LLMGrader flow without
making network calls:

    python examples/llm_provider_examples.py

For live provider calls, install the LLM extra and choose a provider:

    uv pip install -e ".[llm]"
    TRACELENS_LIVE=1 TRACELENS_PROVIDER=openai OPENAI_API_KEY=... python examples/llm_provider_examples.py
    TRACELENS_LIVE=1 TRACELENS_PROVIDER=anthropic ANTHROPIC_API_KEY=... python examples/llm_provider_examples.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from tracelens import LLMGrader, LLMProvider, Task, Transcript


class OpenAIChatProvider(LLMProvider):
    """Minimal OpenAI chat-completions provider.

    The OpenAI SDK is imported only when this provider is constructed, so the
    dry-run path does not require optional dependencies or API keys.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import AsyncOpenAI

        self.model = model
        self._client = AsyncOpenAI()

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        response = await self._client.chat.completions.create(
            model=self.model,
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


class AnthropicMessagesProvider(LLMProvider):
    """Minimal Anthropic Messages provider."""

    def __init__(self, model: str = "claude-3-5-haiku-latest") -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self._client = AsyncAnthropic()

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        response = await self._client.messages.create(
            model=self.model,
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


class RecordedProvider(LLMProvider):
    """Dry-run provider that records prompts and returns canned JSON."""

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name
        self.prompts: list[str] = []

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "instruction_following": 9,
                "feedback": (
                    f"{self.provider_name} dry-run: output follows the requested "
                    "structure and includes actionable detail."
                ),
            }
        )


class InstructionFollowingGrader(LLMGrader):
    """LLMGrader-style quality dimension for instruction following."""

    def __init__(self, provider: LLMProvider, provider_name: str) -> None:
        super().__init__(
            grader_id=f"{provider_name}.instruction_following",
            model=provider_name,
            provider=provider,
        )

    def build_grading_prompt(self, transcript: Transcript, task: Task) -> str:
        return f"""Evaluate the agent output for instruction following.

Task: {task.name}
Input: {json.dumps(task.input_data, indent=2)}
Output: {json.dumps(transcript.final_output, indent=2)}

Score instruction_following from 1 to 10.
Return only JSON:
{{
  "instruction_following": <1-10>,
  "feedback": "<brief explanation>"
}}"""

    def parse_llm_response(
        self,
        response: str,
        task: Task,
    ) -> tuple[bool, float, dict[str, float], str]:
        data = json.loads(response)
        raw_score = float(data["instruction_following"])
        score = raw_score / 10.0
        return (
            score >= 0.7,
            score,
            {"instruction_following": score},
            str(data.get("feedback", "")),
        )


def provider_from_environment(provider_name: str) -> LLMProvider:
    """Return a live provider if requested and configured; otherwise dry-run."""

    live_mode = os.getenv("TRACELENS_LIVE") == "1"

    if live_mode and provider_name == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIChatProvider()
    if live_mode and provider_name == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicMessagesProvider()
    return RecordedProvider(provider_name)


async def run_provider_example(provider_name: str) -> None:
    provider = provider_from_environment(provider_name)
    grader = InstructionFollowingGrader(provider=provider, provider_name=provider_name)

    task = Task(
        task_id=f"{provider_name}-provider-example",
        name="Summarize a support ticket with next steps",
        input_data={
            "ticket": "User cannot connect the app to the hosted database.",
            "required_sections": ["probable cause", "next step", "owner"],
        },
    )
    transcript = Transcript(
        task_id=task.task_id,
        final_output={
            "probable_cause": "DATABASE_URL is missing or points at localhost.",
            "next_step": "Verify the environment variable in the deploy target.",
            "owner": "application maintainer",
        },
    )

    outcome = await grader.grade(transcript, task)
    mode = "live" if not isinstance(provider, RecordedProvider) else "dry-run"
    print(f"{provider_name:<9s} mode={mode:<7s} score={outcome.score:.2f}")
    print(f"  passed  : {outcome.passed}")
    print(f"  feedback: {outcome.feedback}")


async def main() -> None:
    selected = os.getenv("TRACELENS_PROVIDER", "all").lower()
    provider_names = ["openai", "anthropic"] if selected == "all" else [selected]

    for provider_name in provider_names:
        if provider_name not in {"openai", "anthropic"}:
            raise ValueError("TRACELENS_PROVIDER must be 'openai', 'anthropic', or 'all'")
        await run_provider_example(provider_name)


if __name__ == "__main__":
    asyncio.run(main())
