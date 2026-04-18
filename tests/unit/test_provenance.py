"""Unit tests for CitationRecord + format_citations_markdown."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from attune_rag import CitationRecord, CitedSource, format_citations_markdown
from attune_rag.corpus.base import RetrievalEntry
from attune_rag.provenance import build_citation_record
from attune_rag.retrieval import RetrievalHit


def _source(path: str, score: float = 1.0, excerpt: str | None = None) -> CitedSource:
    return CitedSource(template_path=path, category="concepts", score=score, excerpt=excerpt)


def _record(hits: tuple[CitedSource, ...]) -> CitationRecord:
    return CitationRecord(
        query="test query",
        hits=hits,
        retrieved_at=datetime(2026, 4, 17, 12, 0, tzinfo=timezone.utc),
        retriever_name="KeywordRetriever v0.1.0",
    )


def test_empty_hits_renders_no_sources_message() -> None:
    out = format_citations_markdown(_record(()))
    assert "## Sources" in out
    assert "No grounding sources available" in out


def test_single_hit_renders_as_list_item() -> None:
    out = format_citations_markdown(_record((_source("concepts/alpha.md", 0.85),)))
    assert "## Sources" in out
    assert "`concepts/alpha.md`" in out
    assert "concepts" in out
    assert "0.85" in out


def test_multiple_hits_listed_in_order_given() -> None:
    record = _record(
        (
            _source("a.md", 0.9),
            _source("b.md", 0.7),
            _source("c.md", 0.5),
        )
    )
    out = format_citations_markdown(record)
    a_pos = out.index("a.md")
    b_pos = out.index("b.md")
    c_pos = out.index("c.md")
    assert a_pos < b_pos < c_pos


def test_base_url_produces_clickable_links() -> None:
    record = _record((_source("concepts/alpha.md", 0.85),))
    out = format_citations_markdown(record, base_url="https://example.com/docs/")
    assert "[concepts/alpha.md](https://example.com/docs/concepts/alpha.md)" in out


def test_base_url_trailing_slash_handled() -> None:
    record = _record((_source("alpha.md"),))
    with_slash = format_citations_markdown(record, base_url="https://example.com/docs/")
    without_slash = format_citations_markdown(record, base_url="https://example.com/docs")
    assert with_slash == without_slash


def test_excerpt_is_included_when_present() -> None:
    record = _record(
        (
            _source(
                "alpha.md",
                excerpt="The security audit workflow scans for vulnerabilities.",
            ),
        )
    )
    out = format_citations_markdown(record)
    assert "> The security audit workflow scans for vulnerabilities." in out


def test_excerpt_newlines_collapsed() -> None:
    record = _record((_source("alpha.md", excerpt="line one\nline two\nline three"),))
    out = format_citations_markdown(record)
    assert "> line one line two line three" in out


def test_citationrecord_is_frozen_and_hashable() -> None:
    from dataclasses import FrozenInstanceError

    record = _record((_source("alpha.md"),))
    assert isinstance(record, CitationRecord)
    with pytest.raises(FrozenInstanceError):
        record.query = "tampered"  # type: ignore[misc]
    # tuple of frozen dataclasses is hashable
    hash((record.query, record.hits))


def test_build_citation_record_from_hits() -> None:
    entry = RetrievalEntry(
        path="concepts/alpha.md",
        category="concepts",
        content="This is the alpha content.",
    )
    hit = RetrievalHit(entry=entry, score=2.5, match_reason="path:1+content:1")
    now = datetime(2026, 4, 17, tzinfo=timezone.utc)

    record = build_citation_record(
        query="q",
        hits=[hit],
        retriever_name="KeywordRetriever",
        retrieved_at=now,
    )
    assert record.query == "q"
    assert record.retriever_name == "KeywordRetriever"
    assert record.retrieved_at == now
    assert len(record.hits) == 1
    cited = record.hits[0]
    assert cited.template_path == "concepts/alpha.md"
    assert cited.score == 2.5
    assert cited.excerpt == "This is the alpha content."


def test_build_citation_record_excerpt_truncated() -> None:
    entry = RetrievalEntry(
        path="a.md",
        category="x",
        content="x" * 1000,
    )
    hit = RetrievalHit(entry=entry, score=1.0, match_reason="x")
    record = build_citation_record(
        query="q",
        hits=[hit],
        retriever_name="r",
        retrieved_at=datetime.now(timezone.utc),
        excerpt_chars=100,
    )
    assert record.hits[0].excerpt is not None
    assert len(record.hits[0].excerpt) <= 100


def test_build_citation_record_empty_content_no_excerpt() -> None:
    entry = RetrievalEntry(path="a.md", category="x", content="")
    hit = RetrievalHit(entry=entry, score=1.0, match_reason="x")
    record = build_citation_record(
        query="q",
        hits=[hit],
        retriever_name="r",
        retrieved_at=datetime.now(timezone.utc),
    )
    assert record.hits[0].excerpt is None
