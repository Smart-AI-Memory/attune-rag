"""OpenAIProvider — requires ``attune-rag[openai]`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIProvider:
    """Thin async wrapper over OpenAI's chat completions API.

    Lazy-imports ``openai`` so attune-rag installs cleanly
    without the OpenAI SDK.
    """

    name = "openai"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAIProvider requires the [openai] extra. "
                "Install with: pip install 'attune-rag[openai]'"
            ) from exc
        self._client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        cached_prefix: (
            str | None
        ) = None,  # noqa: ARG002 — protocol parity; OpenAI has no equivalent yet
    ) -> str:
        response = await self._client.chat.completions.create(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        return choice.message.content or ""
