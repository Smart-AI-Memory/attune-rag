"""Cache-TTL wire-shape tests for ClaudeProvider.

``ATTUNE_RAG_CACHE_TTL=1h`` extends the prompt-cache window from the
5-minute default to 1 hour by adding ``"ttl": "1h"`` to the
``cache_control`` marker. These tests lock the wire shape at both emit
sites (plain ``generate`` prefix + citations first-document) for the
default and the 1h value, mocking the client — no live API.

See specs/long-cache-ttl-citations/ in the attune umbrella workspace.
The default-shape regression guards live in test_claude.py and
test_claude_citations.py; this file covers the env-driven 1h shape.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from attune_rag.providers.base import CitationDocument
from attune_rag.providers.claude import ClaudeProvider, _cache_control

_EPHEMERAL = {"type": "ephemeral"}
_EPHEMERAL_1H = {"type": "ephemeral", "ttl": "1h"}


# --- _cache_control() helper ------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, _EPHEMERAL),  # unset → default
        ("5m", _EPHEMERAL),  # explicit default
        ("1h", _EPHEMERAL_1H),  # extended
        (" 1H ", _EPHEMERAL_1H),  # case + whitespace insensitive
        ("30m", _EPHEMERAL),  # unsupported value → safe default
        ("", _EPHEMERAL),  # empty → default
    ],
)
def test_cache_control_resolves_from_env(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: dict
) -> None:
    if value is None:
        monkeypatch.delenv("ATTUNE_RAG_CACHE_TTL", raising=False)
    else:
        monkeypatch.setenv("ATTUNE_RAG_CACHE_TTL", value)
    assert _cache_control() == expected


# --- generate() prefix site -------------------------------------------------


def test_generate_prefix_carries_1h_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """With ATTUNE_RAG_CACHE_TTL=1h the cached prefix block carries the
    extended marker; the dynamic tail stays unmarked."""
    monkeypatch.setenv("ATTUNE_RAG_CACHE_TTL", "1h")

    block = MagicMock()
    block.text = "ok"
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    provider = ClaudeProvider(client=client)
    prefix = "STABLE PREFIX\n\n"
    asyncio.run(provider.generate(prefix + "TAIL", cached_prefix=prefix))

    content = client.messages.create.await_args.kwargs["messages"][0]["content"]
    assert content[0]["cache_control"] == _EPHEMERAL_1H
    assert "cache_control" not in content[1]


def test_generate_prefix_defaults_to_5m(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env → the prefix marker is byte-identical to the prior default."""
    monkeypatch.delenv("ATTUNE_RAG_CACHE_TTL", raising=False)

    block = MagicMock()
    block.text = "ok"
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)

    provider = ClaudeProvider(client=client)
    prefix = "STABLE PREFIX\n\n"
    asyncio.run(provider.generate(prefix + "TAIL", cached_prefix=prefix))

    content = client.messages.create.await_args.kwargs["messages"][0]["content"]
    assert content[0]["cache_control"] == _EPHEMERAL


# --- citations first-document site ------------------------------------------


def _docs(*pairs: tuple[str, str]) -> list[CitationDocument]:
    return [CitationDocument(title=t, text=x) for t, x in pairs]


def test_citations_first_doc_carries_1h_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """With 1h set, only the FIRST document block carries the extended
    marker; the i == 0 invariant is preserved."""
    monkeypatch.setenv("ATTUNE_RAG_CACHE_TTL", "1h")
    payload = ClaudeProvider._build_documents_payload(
        _docs(("a.md", "alpha"), ("b.md", "beta"), ("c.md", "gamma"))
    )
    assert payload[0]["cache_control"] == _EPHEMERAL_1H
    for i in range(1, len(payload)):
        assert "cache_control" not in payload[i]


def test_citations_first_doc_defaults_to_5m(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env → first-doc marker is byte-identical to the prior default."""
    monkeypatch.delenv("ATTUNE_RAG_CACHE_TTL", raising=False)
    payload = ClaudeProvider._build_documents_payload(_docs(("a.md", "alpha")))
    assert payload[0]["cache_control"] == _EPHEMERAL
