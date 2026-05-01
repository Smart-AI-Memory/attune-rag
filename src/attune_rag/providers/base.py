"""LLMProvider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """An async LLM provider that consumes a prompt and returns text.

    Implementations live in ``attune_rag.providers.{claude,openai,gemini}``
    behind optional extras. Each lazy-imports its SDK so core
    attune-rag installs cleanly without any provider deps.
    """

    name: str

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        cached_prefix: str | None = None,
    ) -> str:
        """Generate a response from ``prompt``.

        ``cached_prefix`` is an optional stable prefix the
        provider may flag for prompt caching when the
        underlying API supports it. Providers that do not
        support caching must accept the kwarg and ignore it
        — the pipeline always passes it through.
        """
        ...
