"""LiteLLM-backed LLM provider.

Requires ``litellm`` in the ``[llm]`` optional dependency group.
"""

from typing import Any

import litellm

from eval_kit.llm.provider import LLMProvider


class LiteLLMProvider(LLMProvider):
    """Provider backed by LiteLLM (supports 100+ model providers).

    Requires ``litellm`` in the ``[llm]`` optional dependency group.
    """

    def __init__(self, model: str, **default_kwargs: Any) -> None:
        self.model = model
        self.default_kwargs = default_kwargs

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        merged = {**self.default_kwargs, **kwargs}
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **merged,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LiteLLM acompletion failed for model '{self.model}': {exc}"
            ) from exc
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError(
                f"LiteLLM returned null content for model '{self.model}'"
            )
        return content
