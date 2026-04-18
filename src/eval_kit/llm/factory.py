"""Factory for creating LLM providers.

eval-kit ships the ``LLMProvider`` ABC and an ``InMemoryProvider`` for
testing. Real provider integrations (OpenAI, Anthropic, LiteLLM,
Bedrock, anything else) are the user's responsibility — write a
subclass of ``LLMProvider`` that calls your provider's SDK. This keeps
eval-kit decoupled from any specific LLM vendor and avoids shipping a
thin wrapper that duplicates what the vendor SDK already does well.

Canonical user pattern::

    from eval_kit import LLMProvider
    from anthropic import AsyncAnthropic

    class AnthropicProvider(LLMProvider):
        def __init__(self, model: str) -> None:
            self.model = model
            self._client = AsyncAnthropic()

        async def complete(self, prompt: str, **kwargs: Any) -> str:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

Usage::

    from eval_kit.llm.factory import create_provider
    provider = create_provider("in-memory", responses=["mocked"])
"""

from typing import Any

from eval_kit.llm.provider import InMemoryProvider, LLMProvider


def create_provider(model_or_alias: str, **kwargs: Any) -> LLMProvider:
    """Create an LLM provider from an alias.

    Args:
        model_or_alias: Currently only ``"in-memory"`` is supported.
            For real provider calls, subclass ``LLMProvider`` directly —
            eval-kit no longer ships a built-in LiteLLM/OpenAI/Anthropic
            wrapper (see module docstring for the canonical pattern).
        **kwargs: Passed to the provider constructor. For ``"in-memory"``,
            use ``responses=[...]`` to seed canned responses.

    Returns:
        An LLMProvider instance.

    Raises:
        ValueError: If ``model_or_alias`` is anything other than
            ``"in-memory"``. The error message points at the subclassing
            pattern so callers know what to do next.
    """
    if model_or_alias == "in-memory":
        return InMemoryProvider(responses=kwargs.pop("responses", ["mock response"]))

    raise ValueError(
        f"create_provider() only supports 'in-memory'; got {model_or_alias!r}. "
        "For real provider calls, subclass LLMProvider and instantiate it "
        "directly — eval-kit no longer ships a built-in provider wrapper. "
        "See eval_kit.llm.factory docstring for the canonical pattern."
    )
