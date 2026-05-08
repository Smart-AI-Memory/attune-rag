"""Unit tests for ClaudeProvider.generate_with_citations.

Two surfaces are covered:

1. **Payload shape** — the request sent to ``messages.create`` is
   built correctly from ``CitationDocument`` inputs. One document
   block per hit, ``custom_content`` source, ``citations.enabled``
   on each, and the user query as a trailing text block. This
   protects the wire contract; breaking it silently regresses
   citation behavior without any visible error.

2. **Response parsing** — a recorded fixture mirrors the SDK
   0.96.0 ``TextBlock`` + ``CitationContentBlockLocation`` shape,
   exercised through ``ClaudeProvider._parse_cited_response``.
   When the SDK shape shifts, the fixture-vs-parser parity test
   in this file fails first and we know exactly what changed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from attune_rag.providers.base import CitationDocument
from attune_rag.providers.claude import (
    MAX_CITATION_DOCUMENTS,
    ClaudeProvider,
)

_FIXTURE = Path(__file__).resolve().parents[2] / "golden" / "citations_response.json"


def _fake_response_from_fixture() -> SimpleNamespace:
    """Build a SimpleNamespace tree mirroring an Anthropic response.

    SimpleNamespace gives us attribute access matching the SDK's
    pydantic models without requiring the SDK at test time.
    """
    raw = json.loads(_FIXTURE.read_text())
    blocks = []
    for block in raw["content"]:
        cites = []
        for cite in block.get("citations", []) or []:
            cites.append(SimpleNamespace(**cite))
        blocks.append(
            SimpleNamespace(
                type=block["type"],
                text=block.get("text", ""),
                citations=cites or None,
            )
        )
    return SimpleNamespace(content=blocks)


def _docs(*pairs: tuple[str, str]) -> list[CitationDocument]:
    return [CitationDocument(title=t, text=x) for t, x in pairs]


def _provider(response_obj: object) -> tuple[ClaudeProvider, MagicMock]:
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response_obj)
    return ClaudeProvider(client=client), client


# --- Payload shape tests ----------------------------------------------------


def test_documents_payload_shape_one_block_per_doc() -> None:
    """Each CitationDocument should produce one document block with
    ``custom_content`` source + ``citations.enabled``. The payload
    is what protects ``document_index`` alignment with the input
    list.
    """
    docs = _docs(
        ("concepts/a.md", "alpha body"),
        ("concepts/b.md", "beta body"),
    )
    payload = ClaudeProvider._build_documents_payload(docs)
    assert len(payload) == 2
    for i, (title, text) in enumerate(
        [("concepts/a.md", "alpha body"), ("concepts/b.md", "beta body")]
    ):
        assert payload[i]["type"] == "document"
        assert payload[i]["title"] == title
        assert payload[i]["citations"] == {"enabled": True}
        assert payload[i]["source"]["type"] == "content"
        assert payload[i]["source"]["content"] == [{"type": "text", "text": text}]


def test_documents_payload_first_block_carries_cache_control() -> None:
    """``cache_control: ephemeral`` is attached to the FIRST document
    only — that one marker covers the whole document prefix per
    Anthropic's caching semantics (verified by the V2 probe on
    2026-05-08). Subsequent documents in the same request stay
    plain so the wire payload doesn't bloat.
    """
    docs = _docs(
        ("concepts/a.md", "alpha"),
        ("concepts/b.md", "beta"),
        ("concepts/c.md", "gamma"),
    )
    payload = ClaudeProvider._build_documents_payload(docs)
    assert payload[0].get("cache_control") == {"type": "ephemeral"}
    for i in range(1, len(payload)):
        assert "cache_control" not in payload[i], (
            f"document at index {i} unexpectedly carries cache_control; "
            "only the first document should be marked"
        )


def test_documents_payload_single_doc_still_carries_cache_control() -> None:
    """Even with a single document the cache marker is set — that
    single block IS the prefix, and a future second call with the
    same content should hit the cache."""
    docs = _docs(("concepts/only.md", "body"))
    payload = ClaudeProvider._build_documents_payload(docs)
    assert payload[0]["cache_control"] == {"type": "ephemeral"}


def test_generate_with_citations_appends_query_as_trailing_text() -> None:
    response = _fake_response_from_fixture()
    provider, client = _provider(response)
    asyncio.run(
        provider.generate_with_citations(
            documents=_docs(("concepts/a.md", "doc text")),
            query="what does it do?",
            max_tokens=512,
        )
    )
    sent = client.messages.create.await_args.kwargs["messages"]
    assert len(sent) == 1
    content = sent[0]["content"]
    assert content[-1] == {"type": "text", "text": "what does it do?"}
    assert content[0]["type"] == "document"


def test_generate_with_citations_passes_system_prompt_when_given() -> None:
    response = _fake_response_from_fixture()
    provider, client = _provider(response)
    asyncio.run(
        provider.generate_with_citations(
            documents=_docs(("a.md", "x")),
            query="q",
            system="You are an expert.",
        )
    )
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["system"] == "You are an expert."


def test_generate_with_citations_omits_system_when_none() -> None:
    response = _fake_response_from_fixture()
    provider, client = _provider(response)
    asyncio.run(
        provider.generate_with_citations(
            documents=_docs(("a.md", "x")),
            query="q",
        )
    )
    kwargs = client.messages.create.await_args.kwargs
    assert "system" not in kwargs


def test_generate_with_citations_uses_default_model() -> None:
    response = _fake_response_from_fixture()
    provider, client = _provider(response)
    asyncio.run(
        provider.generate_with_citations(
            documents=_docs(("a.md", "x")),
            query="q",
        )
    )
    assert client.messages.create.await_args.kwargs["model"] == ClaudeProvider.DEFAULT_MODEL


def test_generate_with_citations_respects_model_override() -> None:
    response = _fake_response_from_fixture()
    provider, client = _provider(response)
    asyncio.run(
        provider.generate_with_citations(
            documents=_docs(("a.md", "x")),
            query="q",
            model="claude-opus-4-7",
        )
    )
    assert client.messages.create.await_args.kwargs["model"] == "claude-opus-4-7"


def test_generate_with_citations_rejects_empty_documents() -> None:
    provider, _ = _provider(_fake_response_from_fixture())
    with pytest.raises(ValueError, match="at least one document"):
        asyncio.run(provider.generate_with_citations(documents=[], query="q"))


def test_generate_with_citations_rejects_too_many_documents() -> None:
    provider, _ = _provider(_fake_response_from_fixture())
    too_many = _docs(*((f"d{i}.md", "x") for i in range(MAX_CITATION_DOCUMENTS + 1)))
    with pytest.raises(ValueError, match=r"at most \d+ documents"):
        asyncio.run(provider.generate_with_citations(documents=too_many, query="q"))


def test_max_documents_boundary_is_accepted() -> None:
    """Exactly ``MAX_CITATION_DOCUMENTS`` documents must succeed."""
    response = _fake_response_from_fixture()
    provider, client = _provider(response)
    docs = _docs(*((f"d{i}.md", "x") for i in range(MAX_CITATION_DOCUMENTS)))
    asyncio.run(provider.generate_with_citations(documents=docs, query="q"))
    sent_content = client.messages.create.await_args.kwargs["messages"][0]["content"]
    # one block per doc + trailing query text
    assert len(sent_content) == MAX_CITATION_DOCUMENTS + 1


# --- Response parser tests --------------------------------------------------


def test_parse_cited_response_recovers_text_and_citations() -> None:
    parsed = ClaudeProvider._parse_cited_response(_fake_response_from_fixture())
    assert (
        parsed.text
        == "The security audit pipeline runs ruff and bandit in sequence"
        + ", failing on any HIGH-severity finding."
        + " It also surfaces remediation guidance."
    )
    # Two citations from the first two text blocks; third has none.
    assert len(parsed.claim_citations) == 2


def test_parse_cited_response_response_spans_align_to_assembled_text() -> None:
    parsed = ClaudeProvider._parse_cited_response(_fake_response_from_fixture())
    span0 = parsed.claim_citations[0].response_span
    span1 = parsed.claim_citations[1].response_span
    assert span0[0] == 0
    # First block is "The security audit pipeline runs ruff and bandit in sequence"
    assert span0[1] == len("The security audit pipeline runs ruff and bandit in sequence")
    assert span1[0] == span0[1]
    # Span ends at the cumulative length through the second block
    assert span1[1] == span0[1] + len(", failing on any HIGH-severity finding.")


def test_parse_cited_response_extracts_document_attribution() -> None:
    parsed = ClaudeProvider._parse_cited_response(_fake_response_from_fixture())
    for cc in parsed.claim_citations:
        assert cc.document_index == 0
        assert cc.document_title == "concepts/tool-security-audit.md"
        # one-block-per-doc layout always lands at start_block_index=0
        assert cc.cited_block_index == 0


def test_parse_cited_response_handles_no_citations_block_gracefully() -> None:
    """A text block with citations=None or [] must not crash, must not
    contribute to claim_citations, but still contributes its text.
    """
    blocks = [
        SimpleNamespace(type="text", text="alpha", citations=None),
        SimpleNamespace(type="text", text="beta", citations=[]),
    ]
    response = SimpleNamespace(content=blocks)
    parsed = ClaudeProvider._parse_cited_response(response)
    assert parsed.text == "alphabeta"
    assert parsed.claim_citations == ()


def test_parse_cited_response_skips_non_text_blocks() -> None:
    """Tool-use or thinking blocks must be skipped without affecting
    span offsets for subsequent text blocks.
    """
    blocks = [
        SimpleNamespace(type="text", text="hello ", citations=None),
        SimpleNamespace(type="tool_use", id="x", name="t", input={}),
        SimpleNamespace(type="text", text="world", citations=None),
    ]
    response = SimpleNamespace(content=blocks)
    parsed = ClaudeProvider._parse_cited_response(response)
    assert parsed.text == "hello world"


def test_parse_cited_response_defends_against_missing_attributes() -> None:
    """Citation objects with missing optional attributes (e.g. older
    SDK builds, or non-``content_block_location`` types) must
    degrade gracefully — fall back to safe defaults rather than
    crash.
    """
    cite = SimpleNamespace(
        # type-only sentinel; missing start_block_index/document_title/etc.
        type="char_location",
    )
    block = SimpleNamespace(type="text", text="x", citations=[cite])
    response = SimpleNamespace(content=[block])
    parsed = ClaudeProvider._parse_cited_response(response)
    assert len(parsed.claim_citations) == 1
    cc = parsed.claim_citations[0]
    assert cc.document_index == 0
    assert cc.document_title == ""
    assert cc.cited_text == ""
    assert cc.cited_block_index == 0


def test_generate_with_citations_returns_cited_response_end_to_end() -> None:
    response = _fake_response_from_fixture()
    provider, _ = _provider(response)
    result = asyncio.run(
        provider.generate_with_citations(
            documents=_docs(("concepts/tool-security-audit.md", "doc body")),
            query="how does the audit pipeline work?",
        )
    )
    assert result.text.startswith("The security audit pipeline")
    assert len(result.claim_citations) == 2
