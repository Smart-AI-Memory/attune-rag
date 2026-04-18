"""GeminiProvider — requires ``attune-rag[gemini]`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.genai import Client as GenAIClient


class GeminiProvider:
    """Thin async wrapper over Google's genai models API.

    Lazy-imports ``google.genai`` so attune-rag installs cleanly
    without the Gemini SDK.
    """

    name = "gemini"
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: str | None = None,
        client: GenAIClient | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from google.genai import Client
        except ImportError as exc:
            raise RuntimeError(
                "GeminiProvider requires the [gemini] extra. "
                "Install with: pip install 'attune-rag[gemini]'"
            ) from exc
        self._client = Client(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> str:
        config: Any = {"max_output_tokens": max_tokens}
        response = await self._client.aio.models.generate_content(
            model=model or self.DEFAULT_MODEL,
            contents=prompt,
            config=config,
        )
        return getattr(response, "text", "") or ""
