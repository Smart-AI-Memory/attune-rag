"""Unit tests for ClaudeProvider."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_generate_passes_prompt_to_client() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    fake_block = MagicMock()
    fake_block.text = "hello world"
    fake_response = MagicMock()
    fake_response.content = [fake_block]

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=fake_response)

    provider = ClaudeProvider(client=client)
    import asyncio

    result = asyncio.run(provider.generate("what is 2+2?", max_tokens=100))
    assert result == "hello world"
    client.messages.create.assert_awaited_once()
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "what is 2+2?"}]
    assert kwargs["max_tokens"] == 100
    assert kwargs["model"] == ClaudeProvider.DEFAULT_MODEL


def test_generate_respects_model_override() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    fake_block = MagicMock()
    fake_block.text = "ok"
    fake_response = MagicMock()
    fake_response.content = [fake_block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=fake_response)

    provider = ClaudeProvider(client=client)
    import asyncio

    asyncio.run(provider.generate("q", model="claude-opus-4-7"))
    assert client.messages.create.await_args.kwargs["model"] == "claude-opus-4-7"


def test_generate_concatenates_multiple_text_blocks() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    b1 = MagicMock()
    b1.text = "one "
    b2 = MagicMock()
    b2.text = "two"
    response = MagicMock()
    response.content = [b1, b2]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    import asyncio

    provider = ClaudeProvider(client=client)
    assert asyncio.run(provider.generate("q")) == "one two"


def test_missing_sdk_raises_helpful_error() -> None:
    """Uses sys.modules patching per CLAUDE.md MetaPathFinder lesson."""
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key == "anthropic" or key.startswith("anthropic."):
            saved[key] = sys.modules.pop(key)
    # Also purge the already-imported adapter so its lazy import retries.
    for key in list(sys.modules):
        if key == "attune_rag.providers.claude":
            saved[key] = sys.modules.pop(key)

    class Blocker:
        def find_module(self, name, path=None):  # noqa: ARG002
            if name == "anthropic" or name.startswith("anthropic."):
                return self
            return None

        def load_module(self, name):
            raise ImportError(f"BLOCKED: {name}")

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    try:
        from attune_rag.providers.claude import ClaudeProvider

        with pytest.raises(RuntimeError, match=r"\[claude\] extra"):
            ClaudeProvider()
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(saved)


def test_provider_name() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    assert ClaudeProvider.name == "claude"
