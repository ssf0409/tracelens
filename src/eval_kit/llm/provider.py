"""LLM provider abstraction.

Providers handle the actual LLM API call, allowing graders to be
decoupled from specific LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send a prompt to the LLM and return the text response."""
        ...


class InMemoryProvider(LLMProvider):
    """Testing provider that returns canned responses.

    Cycles through the provided responses list and records all prompts.
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []
        self._index = 0

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        return response


class LiteLLMProvider(LLMProvider):
    """Provider backed by LiteLLM (supports 100+ model providers).

    Requires ``litellm`` in the ``[llm]`` optional dependency group.
    """

    def __init__(self, model: str, **default_kwargs: Any) -> None:
        self.model = model
        self.default_kwargs = default_kwargs

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        import litellm

        merged = {**self.default_kwargs, **kwargs}
        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **merged,
        )
        return response.choices[0].message.content
