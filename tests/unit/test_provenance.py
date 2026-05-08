"""Unit tests for CitationRecord + format_citations_markdown."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from attune_rag import (
    CitationRecord,
    CitedSource,
    format_citations_markdown,
)
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


# --- ClaimCitation -----------------------------------------------------------


def _claim(
    span: tuple[int, int] = (0, 10),
    document_index: int = 0,
    document_title: str = "concepts/foo.md",
    cited_text: str = "alpha",
    cited_block_index: int = 0,
):
    from attune_rag import ClaimCitation as _ClaimCitation

    return _ClaimCitation(
        response_span=span,
        document_index=document_index,
        document_title=document_title,
        cited_text=cited_text,
        cited_block_index=cited_block_index,
    )


def test_claim_citation_construction_all_fields() -> None:
    cc = _claim(
        span=(12, 45),
        document_index=2,
        document_title="concepts/security-audit.md",
        cited_text="ruff and bandit run in sequence",
        cited_block_index=0,
    )
    assert cc.response_span == (12, 45)
    assert cc.document_index == 2
    assert cc.document_title == "concepts/security-audit.md"
    assert cc.cited_text == "ruff and bandit run in sequence"
    assert cc.cited_block_index == 0


def test_claim_citation_default_block_index() -> None:
    from attune_rag import ClaimCitation as _ClaimCitation

    cc = _ClaimCitation(
        response_span=(0, 5),
        document_index=0,
        document_title="t.md",
        cited_text="x",
    )
    assert cc.cited_block_index == 0


def test_claim_citation_equality_and_frozen() -> None:
    a = _claim()
    b = _claim()
    assert a == b
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.cited_text = "mutated"  # type: ignore[misc]


def test_claim_citation_distinct_when_any_field_differs() -> None:
    base = _claim()
    assert base != _claim(span=(0, 11))
    assert base != _claim(document_index=1)
    assert base != _claim(document_title="other.md")
    assert base != _claim(cited_text="beta")
    assert base != _claim(cited_block_index=2)


def test_claim_citation_hashable_for_use_in_sets() -> None:
    a = _claim()
    b = _claim()
    c = _claim(document_index=99)
    assert {a, b, c} == {a, c}


def test_claim_citation_re_exported_from_top_level() -> None:
    import attune_rag

    assert "ClaimCitation" in attune_rag.__all__
    assert hasattr(attune_rag, "ClaimCitation")


# --- format_claim_citations_markdown -----------------------------------------


def _cc(
    span: tuple[int, int],
    document_index: int = 0,
    document_title: str = "concepts/foo.md",
    cited_text: str = "source quote",
):
    from attune_rag import ClaimCitation as _ClaimCitation

    return _ClaimCitation(
        response_span=span,
        document_index=document_index,
        document_title=document_title,
        cited_text=cited_text,
    )


def test_format_claim_citations_empty_returns_text_with_note() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    out = fmt("Hello world.", citations=())
    assert "No claim-level citations" in out
    assert "Hello world." in out


def test_format_claim_citations_single_citation_appends_marker_and_footnote() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Ruff and bandit run in sequence."
    cc = _cc(
        span=(0, len(text)),
        document_index=0,
        document_title="concepts/security.md",
        cited_text="ruff and bandit run in sequence",
    )
    out = fmt(text, [cc])
    assert "[^1]" in out
    # marker spliced at end of cited span — i.e. before the trailing newline
    assert text + "[^1]" in out
    assert '[^1]: `concepts/security.md` — "ruff and bandit run in sequence"' in out


def test_format_claim_citations_same_document_shares_footnote_number() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Alpha. Beta."
    a = _cc(span=(0, 6), document_title="d.md", cited_text="alpha")
    b = _cc(span=(7, 12), document_title="d.md", cited_text="beta")
    out = fmt(text, [a, b])
    # Same document => same footnote number => only ONE definition line
    assert out.count("[^1]:") == 1
    assert "[^2]" not in out
    # But TWO in-text markers
    assert out.count("[^1]") == 3  # 2 in-text + 1 definition


def test_format_claim_citations_distinct_documents_get_separate_footnotes() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Alpha. Beta."
    a = _cc(span=(0, 6), document_index=0, document_title="d1.md", cited_text="a")
    b = _cc(span=(7, 12), document_index=1, document_title="d2.md", cited_text="b")
    out = fmt(text, [a, b])
    assert "[^1]:" in out
    assert "[^2]:" in out
    assert "`d1.md`" in out
    assert "`d2.md`" in out


def test_format_claim_citations_marker_order_follows_span_start() -> None:
    """In-text markers should appear in narrative order, not citation
    iteration order. Pass citations out of order; output must still
    show [^1] before [^2] in the text.
    """
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Alpha. Beta."
    later = _cc(span=(7, 12), document_index=1, document_title="d2.md", cited_text="b")
    earlier = _cc(span=(0, 6), document_index=0, document_title="d1.md", cited_text="a")
    out = fmt(text, [later, earlier])
    # [^1] must appear before [^2] in the rendered text
    assert out.index("[^1]") < out.index("[^2]")


def test_format_claim_citations_base_url_produces_clickable_links() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Hello."
    cc = _cc(span=(0, 6), document_title="d.md", cited_text="hi")
    out = fmt(text, [cc], base_url="https://example.com")
    assert "[`d.md`](https://example.com/d.md)" in out


def test_format_claim_citations_base_url_trailing_slash_handled() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Hello."
    cc = _cc(span=(0, 6), document_title="d.md", cited_text="hi")
    out = fmt(text, [cc], base_url="https://example.com/")
    assert "https://example.com/d.md" in out
    assert "https://example.com//d.md" not in out


def test_format_claim_citations_strips_newlines_in_cited_text() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Hello."
    cc = _cc(span=(0, 6), document_title="d.md", cited_text="line1\nline2  ")
    out = fmt(text, [cc])
    assert "line1\nline2" not in out
    assert "line1 line2" in out


def test_format_claim_citations_missing_title_renders_placeholder() -> None:
    from attune_rag import format_claim_citations_markdown as fmt

    text = "Hello."
    cc = _cc(span=(0, 6), document_title="", cited_text="x")
    out = fmt(text, [cc])
    assert "(untitled)" in out


def test_format_claim_citations_helper_re_exported_from_top_level() -> None:
    import attune_rag

    assert "format_claim_citations_markdown" in attune_rag.__all__
    assert hasattr(attune_rag, "format_claim_citations_markdown")
