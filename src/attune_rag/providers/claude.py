"""ClaudeProvider — requires ``attune-rag[claude]`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


class ClaudeProvider:
    """Thin async wrapper over Anthropic's Messages API.

    Lazy-imports ``anthropic`` so attune-rag installs cleanly
    without the Claude SDK.
    """

    name = "claude"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "ClaudeProvider requires the [claude] extra. "
                "Install with: pip install 'attune-rag[claude]'"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        cached_prefix: str | None = None,
    ) -> str:
        if cached_prefix:
            # Two-block message: stable context (cached) + dynamic tail
            tail = prompt[len(cached_prefix) :]
            content = [
                {
                    "type": "text",
                    "text": cached_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": tail},
            ]
        else:
            content = prompt

        response = await self._client.messages.create(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        chunks: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks)
