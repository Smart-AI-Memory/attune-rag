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
    """Simulate anthropic not being installed via sys.modules sentinel.

    Setting ``sys.modules[name] = None`` causes Python's import
    machinery to raise ImportError on the next import of ``name``.
    Works across Python 3.10-3.13 (the deprecated MetaPathFinder
    API stopped firing in 3.12+).
    """
    saved: dict[str, object] = {}
    # Purge real anthropic modules and the already-imported adapter
    # so its lazy `from anthropic import ...` retries.
    for key in list(sys.modules):
        if key in {"anthropic", "attune_rag.providers.claude"} or key.startswith("anthropic."):
            saved[key] = sys.modules.pop(key)

    # Sentinel: forces ImportError on next `import anthropic`
    sys.modules["anthropic"] = None  # type: ignore[assignment]

    try:
        from attune_rag.providers.claude import ClaudeProvider

        with pytest.raises(RuntimeError, match=r"\[claude\] extra"):
            ClaudeProvider()
    finally:
        sys.modules.pop("anthropic", None)
        sys.modules.update(saved)


def test_provider_name() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    assert ClaudeProvider.name == "claude"


def test_cached_prefix_splits_into_two_blocks_with_cache_control() -> None:
    """When a cached_prefix is supplied, the provider must send a
    two-block message: a flagged prefix and the dynamic tail.

    This is the contract that powers Anthropic prompt caching;
    breaking it silently regresses cache hit rate without any
    visible error, so we lock the message shape in a unit test.
    """
    from attune_rag.providers.claude import ClaudeProvider

    block = MagicMock()
    block.text = "ok"
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    provider = ClaudeProvider(client=client)
    prefix = "STABLE PREFIX\n\n"
    tail = "DYNAMIC TAIL"
    prompt = prefix + tail

    import asyncio

    asyncio.run(provider.generate(prompt, cached_prefix=prefix))

    sent = client.messages.create.await_args.kwargs["messages"]
    assert sent == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": tail},
            ],
        }
    ]


def test_no_cached_prefix_sends_plain_string_content() -> None:
    """Without a cached_prefix, the provider falls back to the
    pre-caching single-string content shape so non-cached call
    sites keep their original wire payload.
    """
    from attune_rag.providers.claude import ClaudeProvider

    block = MagicMock()
    block.text = "ok"
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    provider = ClaudeProvider(client=client)
    import asyncio

    asyncio.run(provider.generate("plain prompt"))
    assert client.messages.create.await_args.kwargs["messages"] == [
        {"role": "user", "content": "plain prompt"}
    ]
