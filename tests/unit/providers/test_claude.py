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


# --- model tiers + fable handling (specs/fable-model-tiers, task 2) ---


def _client_returning(response: MagicMock) -> MagicMock:
    """Mock client with both namespaces wired to the same response."""
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    client.beta.messages.create = AsyncMock(return_value=response)
    return client


def _text_response(text: str = "ok") -> MagicMock:
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    response.stop_reason = "end_turn"
    return response


def test_default_model_is_capable_tier(monkeypatch) -> None:
    from attune_rag.providers.claude import ClaudeProvider

    monkeypatch.delenv("ATTUNE_MODEL_CAPABLE", raising=False)
    client = _client_returning(_text_response())
    import asyncio

    asyncio.run(ClaudeProvider(client=client).generate("q"))
    assert client.messages.create.await_args.kwargs["model"] == "claude-sonnet-5"


def test_capable_env_pin_changes_default_without_reimport(monkeypatch) -> None:
    from attune_rag.providers.claude import ClaudeProvider

    monkeypatch.setenv("ATTUNE_MODEL_CAPABLE", "claude-sonnet-4-6")
    client = _client_returning(_text_response())
    import asyncio

    asyncio.run(ClaudeProvider(client=client).generate("q"))
    assert client.messages.create.await_args.kwargs["model"] == "claude-sonnet-4-6"


def test_fable_model_routes_to_beta_namespace_with_extras() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    client = _client_returning(_text_response("fable says"))
    import asyncio

    result = asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-fable-5"))
    assert result == "fable says"
    client.messages.create.assert_not_awaited()
    kwargs = client.beta.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-fable-5"
    assert kwargs["betas"] == ["server-side-fallback-2026-06-01"]
    assert kwargs["extra_body"] == {"fallbacks": [{"model": "claude-opus-4-8"}]}


def test_capable_env_pinned_to_fable_routes_to_beta(monkeypatch) -> None:
    """The beta switch keys off the *effective* model, not the tier name."""
    from attune_rag.providers.claude import ClaudeProvider

    monkeypatch.setenv("ATTUNE_MODEL_CAPABLE", "claude-fable-5")
    client = _client_returning(_text_response())
    import asyncio

    asyncio.run(ClaudeProvider(client=client).generate("q"))
    client.messages.create.assert_not_awaited()
    assert client.beta.messages.create.await_args.kwargs["model"] == "claude-fable-5"


def test_non_fable_call_never_touches_beta_namespace() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    client = _client_returning(_text_response())
    import asyncio

    asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-sonnet-5"))
    client.beta.messages.create.assert_not_awaited()
    kwargs = client.messages.create.await_args.kwargs
    assert "betas" not in kwargs
    assert "extra_body" not in kwargs


def test_fable_refusal_raises_model_refusal_error() -> None:
    from attune_rag.model_tiers import ModelRefusalError
    from attune_rag.providers.claude import ClaudeProvider

    response = _text_response()
    response.stop_reason = "refusal"
    response.stop_details = MagicMock()
    response.stop_details.category = "harmful_content"
    response.stop_details.explanation = "declined"
    client = _client_returning(response)
    import asyncio

    with pytest.raises(ModelRefusalError) as excinfo:
        asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-fable-5"))
    assert excinfo.value.category == "harmful_content"
    assert excinfo.value.explanation == "declined"


def test_fable_refusal_with_dict_stop_details() -> None:
    from attune_rag.model_tiers import ModelRefusalError
    from attune_rag.providers.claude import ClaudeProvider

    response = _text_response()
    response.stop_reason = "refusal"
    response.stop_details = {"category": "other", "explanation": "no"}
    client = _client_returning(response)
    import asyncio

    with pytest.raises(ModelRefusalError) as excinfo:
        asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-fable-5"))
    assert excinfo.value.category == "other"


def test_non_fable_refusal_stop_reason_is_not_checked() -> None:
    """Pre-tier behavior preserved: sonnet/haiku responses are read as-is."""
    from attune_rag.providers.claude import ClaudeProvider

    response = _text_response("text anyway")
    response.stop_reason = "refusal"
    client = _client_returning(response)
    import asyncio

    result = asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-sonnet-5"))
    assert result == "text anyway"


def test_fable_400_carries_retention_hint() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    class Fake400(Exception):
        status_code = 400

    client = MagicMock()
    client.beta.messages.create = AsyncMock(
        side_effect=Fake400("invalid_request_error: bad payload")
    )
    import asyncio

    with pytest.raises(Fake400) as excinfo:
        asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-fable-5"))
    assert "30-day org data retention" in str(excinfo.value)
    assert "invalid_request_error: bad payload" in str(excinfo.value)


def test_fable_non_400_error_passes_through_unhinted() -> None:
    from attune_rag.providers.claude import ClaudeProvider

    class Fake500(Exception):
        status_code = 500

    client = MagicMock()
    client.beta.messages.create = AsyncMock(side_effect=Fake500("overloaded"))
    import asyncio

    with pytest.raises(Fake500) as excinfo:
        asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-fable-5"))
    assert "retention" not in str(excinfo.value)


def test_explicit_model_literal_passed_through_untouched(monkeypatch) -> None:
    from attune_rag.providers.claude import ClaudeProvider

    monkeypatch.setenv("ATTUNE_MODEL_CAPABLE", "claude-fable-5")
    client = _client_returning(_text_response())
    import asyncio

    asyncio.run(ClaudeProvider(client=client).generate("q", model="claude-haiku-4-5"))
    client.beta.messages.create.assert_not_awaited()
    assert client.messages.create.await_args.kwargs["model"] == "claude-haiku-4-5"


def test_citations_with_fable_model_routes_to_beta() -> None:
    from attune_rag.providers.base import CitationDocument
    from attune_rag.providers.claude import ClaudeProvider

    response = MagicMock()
    response.content = []
    response.stop_reason = "end_turn"
    client = _client_returning(response)
    import asyncio

    asyncio.run(
        ClaudeProvider(client=client).generate_with_citations(
            [CitationDocument(title="t", text="body")],
            "q",
            model="claude-fable-5",
        )
    )
    client.messages.create.assert_not_awaited()
    kwargs = client.beta.messages.create.await_args.kwargs
    assert kwargs["betas"] == ["server-side-fallback-2026-06-01"]


def test_default_model_alias_still_exists() -> None:
    """Deprecated alias kept for external references; equals the capable tier
    default under a clean env (import-time snapshot)."""
    from attune_rag.providers.claude import ClaudeProvider

    assert ClaudeProvider.DEFAULT_MODEL == "claude-sonnet-5"
