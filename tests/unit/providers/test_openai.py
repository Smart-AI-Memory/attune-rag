"""Unit tests for OpenAIProvider."""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _build_response(text: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def test_generate_passes_prompt_to_client() -> None:
    from attune_rag.providers.openai import OpenAIProvider

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_build_response("hello"))
    provider = OpenAIProvider(client=client)
    result = asyncio.run(provider.generate("prompt"))
    assert result == "hello"
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]
    assert kwargs["model"] == OpenAIProvider.DEFAULT_MODEL


def test_generate_respects_model_override() -> None:
    from attune_rag.providers.openai import OpenAIProvider

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_build_response("ok"))
    provider = OpenAIProvider(client=client)
    asyncio.run(provider.generate("q", model="gpt-4-turbo"))
    assert client.chat.completions.create.await_args.kwargs["model"] == "gpt-4-turbo"


def test_missing_sdk_raises_helpful_error() -> None:
    """sys.modules sentinel — works across Python 3.10-3.13."""
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key in {"openai", "attune_rag.providers.openai"} or key.startswith("openai."):
            saved[key] = sys.modules.pop(key)

    sys.modules["openai"] = None  # type: ignore[assignment]

    try:
        from attune_rag.providers.openai import OpenAIProvider

        with pytest.raises(RuntimeError, match=r"\[openai\] extra"):
            OpenAIProvider()
    finally:
        sys.modules.pop("openai", None)
        sys.modules.update(saved)


def test_provider_name() -> None:
    from attune_rag.providers.openai import OpenAIProvider

    assert OpenAIProvider.name == "openai"
