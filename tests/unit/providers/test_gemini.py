"""Unit tests for GeminiProvider."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_generate_passes_prompt_to_client() -> None:
    from attune_rag.providers.gemini import GeminiProvider

    response = MagicMock()
    response.text = "gemini says hi"
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)

    provider = GeminiProvider(client=client)
    result = asyncio.run(provider.generate("prompt", max_tokens=500))
    assert result == "gemini says hi"
    kwargs = client.aio.models.generate_content.await_args.kwargs
    assert kwargs["contents"] == "prompt"
    assert kwargs["model"] == GeminiProvider.DEFAULT_MODEL
    assert kwargs["config"]["max_output_tokens"] == 500


def test_generate_respects_model_override() -> None:
    from attune_rag.providers.gemini import GeminiProvider

    response = MagicMock()
    response.text = "ok"
    client = MagicMock()
    client.aio.models.generate_content = AsyncMock(return_value=response)

    provider = GeminiProvider(client=client)
    asyncio.run(provider.generate("q", model="gemini-1.5-pro"))
    assert client.aio.models.generate_content.await_args.kwargs["model"] == "gemini-1.5-pro"


def test_missing_sdk_raises_helpful_error() -> None:
    """sys.modules sentinel — works across Python 3.10-3.13."""
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key in {"google.genai", "attune_rag.providers.gemini"} or key.startswith(
            "google.genai."
        ):
            saved[key] = sys.modules.pop(key)

    sys.modules["google.genai"] = None  # type: ignore[assignment]

    try:
        from attune_rag.providers.gemini import GeminiProvider

        with pytest.raises(RuntimeError, match=r"\[gemini\] extra"):
            GeminiProvider()
    finally:
        sys.modules.pop("google.genai", None)
        sys.modules.update(saved)


def test_provider_name() -> None:
    from attune_rag.providers.gemini import GeminiProvider

    assert GeminiProvider.name == "gemini"
