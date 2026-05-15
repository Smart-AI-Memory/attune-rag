"""Pipeline tests for the native-citations code path.

Covers all four rows of the behavior matrix from
``specs/rag-native-citations/design.md``:

| use_native_citations | provider supports | hits empty | path                         |
|----------------------|-------------------|------------|------------------------------|
| False                | —                 | —          | legacy prompt-assembly       |
| True                 | True              | False      | native citations             |
| True                 | True              | True       | fallback prompt (no docs)    |
| True                 | False             | —          | warning + legacy fallback    |

Plus a parity test guarding against retrieval-time drift between
the two paths: same query → same ``RagResult.citation`` and
``context``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

import pytest

from attune_rag import (
    ClaimCitation,
    RagPipeline,
    RetrievalEntry,
)
from attune_rag.corpus.base import CorpusProtocol
from attune_rag.providers.base import CitationDocument, CitedResponse

# --- fixtures ---------------------------------------------------------------


class FakeCorpus(CorpusProtocol):
    def __init__(self, entries: Iterable[RetrievalEntry]) -> None:
        self._entries = {e.path: e for e in entries}

    def entries(self) -> Iterable[RetrievalEntry]:
        return tuple(self._entries.values())

    def get(self, path: str) -> RetrievalEntry | None:
        return self._entries.get(path)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def version(self) -> str:
        return "0"


@pytest.fixture
def corpus() -> FakeCorpus:
    return FakeCorpus(
        [
            RetrievalEntry(
                path="concepts/security-audit.md",
                category="concepts",
                content="Security audit scans for vulnerabilities.",
                summary="Run a security audit",
            ),
            RetrievalEntry(
                path="concepts/smart-test.md",
                category="concepts",
                content="Smart test generates tests for untested code.",
                summary="Generate tests",
            ),
        ]
    )


class StubCitationsProvider:
    """Stub provider that returns a canned ``CitedResponse``.

    Records inputs so tests can assert on payload shape. Does
    NOT touch the network. Mirrors the duck type expected by
    ``RagPipeline.run_and_generate`` — sets the
    ``supports_native_citations`` flag and implements both
    ``generate`` and ``generate_with_citations``.
    """

    name = "stub-claude"
    supports_native_citations = True

    def __init__(self, cited_response: CitedResponse | None = None) -> None:
        self._cited = cited_response or CitedResponse(
            text="Audits scan code.",
            claim_citations=(
                ClaimCitation(
                    response_span=(0, 18),
                    document_index=0,
                    document_title="concepts/security-audit.md",
                    cited_text="Security audit scans for vulnerabilities.",
                ),
            ),
        )
        self.citations_calls: list[dict] = []
        self.generate_calls: list[dict] = []

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        cached_prefix: str | None = None,
    ) -> str:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "cached_prefix": cached_prefix,
            }
        )
        return "[stub generate text]"

    async def generate_with_citations(
        self,
        documents: list[CitationDocument],
        query: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> CitedResponse:
        self.citations_calls.append(
            {
                "documents": list(documents),
                "query": query,
                "system": system,
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        return self._cited


class NoCitationsProvider:
    """Stub for the 'provider does not support citations' row.

    Implements ``generate`` only; ``supports_native_citations`` is
    False. The pipeline should detect this and fall back to the
    legacy path WITHOUT calling ``generate_with_citations``.
    """

    name = "stub-gemini"
    supports_native_citations = False

    def __init__(self) -> None:
        self.generate_calls: list[dict] = []

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        cached_prefix: str | None = None,
    ) -> str:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "model": model,
                "max_tokens": max_tokens,
                "cached_prefix": cached_prefix,
            }
        )
        return "[stub legacy text]"


# --- Row 1: use_native_citations=False --------------------------------------


def test_legacy_path_when_use_native_citations_false(corpus: FakeCorpus) -> None:
    """Default path: hits exist, native off, plain generate is called."""
    pipeline = RagPipeline(corpus=corpus)
    provider = StubCitationsProvider()
    response, result = asyncio.run(pipeline.run_and_generate("security audit", provider=provider))
    assert response == "[stub generate text]"
    assert result.used_native_citations is False
    assert result.claim_citations == ()
    assert result.augmented_prompt  # non-empty: legacy path renders it
    assert provider.citations_calls == []  # citations API not called
    assert len(provider.generate_calls) == 1


# --- Row 2: True + provider supports + hits exist ---------------------------


def test_native_path_when_supported_and_hits_exist(corpus: FakeCorpus) -> None:
    pipeline = RagPipeline(corpus=corpus)
    provider = StubCitationsProvider()
    response, result = asyncio.run(
        pipeline.run_and_generate(
            "security audit",
            provider=provider,
            use_native_citations=True,
        )
    )
    assert response == "Audits scan code."
    assert result.used_native_citations is True
    assert len(result.claim_citations) == 1
    assert result.claim_citations[0].document_title == "concepts/security-audit.md"
    # Legacy generate must NOT be called on this path
    assert provider.generate_calls == []
    assert len(provider.citations_calls) == 1
    # Documents are built from hit content (not the truncated excerpt)
    sent_docs = provider.citations_calls[0]["documents"]
    assert all(isinstance(d, CitationDocument) for d in sent_docs)
    assert any("Security audit scans for vulnerabilities." in d.text for d in sent_docs)


def test_native_path_populates_rag_result_fields(corpus: FakeCorpus) -> None:
    pipeline = RagPipeline(corpus=corpus)
    provider = StubCitationsProvider()
    _, result = asyncio.run(
        pipeline.run_and_generate(
            "security audit",
            provider=provider,
            use_native_citations=True,
        )
    )
    # citation (retrieval-level) is still populated for parity
    assert result.citation.hits
    # confidence + elapsed are populated
    assert result.confidence > 0
    assert result.elapsed_ms >= 0
    # context is populated for eval parity (same content the model saw)
    assert "Security audit" in result.context
    # augmented_prompt is empty: documents went over the wire as blocks,
    # not as a rendered prompt
    assert result.augmented_prompt == ""
    assert result.fallback_used is False


def test_native_path_passes_model_and_max_tokens_through(corpus: FakeCorpus) -> None:
    pipeline = RagPipeline(corpus=corpus)
    provider = StubCitationsProvider()
    asyncio.run(
        pipeline.run_and_generate(
            "security audit",
            provider=provider,
            use_native_citations=True,
            model="claude-opus-4-7",
            max_tokens=4096,
        )
    )
    call = provider.citations_calls[0]
    assert call["model"] == "claude-opus-4-7"
    assert call["max_tokens"] == 4096


# --- Row 3: True + provider supports + hits empty ---------------------------


def test_native_path_with_empty_hits_uses_fallback_prompt(corpus: FakeCorpus) -> None:
    """No retrievable docs → no citations call; fallback prompt + legacy
    generate. ``used_native_citations`` stays False because the
    citations API was never engaged.
    """
    pipeline = RagPipeline(corpus=corpus)
    provider = StubCitationsProvider()
    response, result = asyncio.run(
        pipeline.run_and_generate(
            "zzzzz asdfgh qwerty unrelated",
            provider=provider,
            use_native_citations=True,
        )
    )
    assert response == "[stub generate text]"
    assert result.used_native_citations is False
    assert result.claim_citations == ()
    assert result.fallback_used is True
    assert result.citation.hits == ()
    assert "No grounding context" in result.augmented_prompt
    # Citations API must NOT be called when there are no docs
    assert provider.citations_calls == []
    assert len(provider.generate_calls) == 1


# --- Row 4: True + provider does NOT support -------------------------------


def test_native_path_falls_back_when_provider_unsupported(
    corpus: FakeCorpus,
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline = RagPipeline(corpus=corpus)
    provider = NoCitationsProvider()
    with caplog.at_level(logging.WARNING, logger="attune_rag.pipeline"):
        response, result = asyncio.run(
            pipeline.run_and_generate(
                "security audit",
                provider=provider,
                use_native_citations=True,
            )
        )
    assert response == "[stub legacy text]"
    assert result.used_native_citations is False
    assert result.claim_citations == ()
    # Legacy path: augmented_prompt is rendered, generate was called.
    assert result.augmented_prompt
    assert len(provider.generate_calls) == 1


# --- Parity test (guards against drift) -------------------------------------


def test_retrieval_parity_between_legacy_and_native_paths(corpus: FakeCorpus) -> None:
    """Both paths must produce the same retrieval-level provenance.

    The two paths only diverge in *generation*. If retrieval drifts
    (e.g. someone tweaks query expansion in ``run`` but not in
    ``_run_native_citations``), evaluation results stop being
    comparable. This test pins that invariant.
    """
    pipeline = RagPipeline(corpus=corpus)
    provider = StubCitationsProvider()

    _, legacy = asyncio.run(
        pipeline.run_and_generate("security audit", provider=provider, use_native_citations=False)
    )
    _, native = asyncio.run(
        pipeline.run_and_generate("security audit", provider=provider, use_native_citations=True)
    )

    # Same hits, same paths, same scores.
    assert tuple(h.template_path for h in legacy.citation.hits) == tuple(
        h.template_path for h in native.citation.hits
    )
    assert tuple(h.score for h in legacy.citation.hits) == tuple(
        h.score for h in native.citation.hits
    )
    # Same context rendered for evaluators.
    assert legacy.context == native.context
    # Same confidence calculation.
    assert legacy.confidence == native.confidence
    # And — importantly — different generation behavior is reflected in
    # the flags so consumers can branch correctly.
    assert legacy.used_native_citations is False
    assert native.used_native_citations is True


def test_native_path_documents_carry_full_content_not_truncated_excerpt(
    corpus: FakeCorpus,
) -> None:
    """``CitationRecord.hits[].excerpt`` is truncated to 200 chars by
    default. The native-citations path must send the full hit content
    via ``CitationDocument.text`` so the model can cite spans beyond
    the excerpt window. This test guards against a subtle bug where
    someone "helpfully" wires the excerpt through.
    """
    long_content = "X" * 500 + " UNIQUE_TAIL_MARKER " + "Y" * 500
    bigger_corpus = FakeCorpus(
        [
            RetrievalEntry(
                path="concepts/long.md",
                category="concepts",
                content=long_content,
                summary="A long doc",
            )
        ]
    )
    pipeline = RagPipeline(corpus=bigger_corpus)
    provider = StubCitationsProvider()
    asyncio.run(
        pipeline.run_and_generate(
            "long doc",
            provider=provider,
            use_native_citations=True,
        )
    )
    sent = provider.citations_calls[0]["documents"]
    assert any("UNIQUE_TAIL_MARKER" in d.text for d in sent)


# --- Issue #14: cache_control + citations on the same request --------------


def test_pipeline_request_carries_both_cache_control_and_citations(
    corpus: FakeCorpus,
) -> None:
    """End-to-end: pipeline → real ClaudeProvider → mocked SDK client.

    Cache_control and the Citations API are independent cost-perf
    levers; existing tests cover each at the helper level
    (``_build_documents_payload``) but nothing asserts they BOTH
    survive to the actual ``messages.create`` payload on a single
    request driven through the pipeline. A regression in either
    (e.g., a refactor that swaps payload builders, or a flag that
    silently disables one) would otherwise pass green.
    """
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from attune_rag.providers.claude import ClaudeProvider

    fixture_path = Path(__file__).resolve().parents[1] / "golden" / "citations_response.json"
    raw = json.loads(fixture_path.read_text())
    blocks = []
    for block in raw["content"]:
        cites = [SimpleNamespace(**c) for c in (block.get("citations") or [])]
        blocks.append(
            SimpleNamespace(
                type=block["type"],
                text=block.get("text", ""),
                citations=cites or None,
            )
        )
    response_obj = SimpleNamespace(content=blocks)

    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response_obj)
    provider = ClaudeProvider(client=client)

    pipeline = RagPipeline(corpus=corpus)
    asyncio.run(
        pipeline.run_and_generate(
            "security audit",
            provider=provider,
            use_native_citations=True,
        )
    )

    sent_messages = client.messages.create.await_args.kwargs["messages"]
    assert len(sent_messages) == 1
    content = sent_messages[0]["content"]
    doc_blocks = [b for b in content if b.get("type") == "document"]
    assert doc_blocks, "pipeline did not send any document blocks"

    # Citations API enabled on every document block.
    for i, block in enumerate(doc_blocks):
        assert block.get("citations") == {"enabled": True}, (
            f"document block {i} missing citations:enabled — " f"got {block.get('citations')!r}"
        )

    # cache_control on at least one document block (the provider's
    # current strategy is first-block-only, but the assertion is
    # deliberately broader so a future change to mark every block
    # — or any subset — does not break this test).
    cached = [b for b in doc_blocks if b.get("cache_control") == {"type": "ephemeral"}]
    assert cached, (
        "no document block carries cache_control={'type':'ephemeral'} — "
        "prompt caching has silently regressed on the citations path"
    )
