"""Provenance records + markdown formatting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ClaimCitation:
    """One claim-level citation produced by the Anthropic Citations API.

    Unlike :class:`CitedSource` (which is retrieval-time and
    document-level), a ``ClaimCitation`` is asserted by the model
    at generation time and points at the exact span of source
    content the model attributes a specific response span to.

    Attributes:
        response_span: ``(start, end)`` char offsets in the
            assembled response text identifying the cited claim.
        document_index: Index into the documents list passed to
            the provider — i.e. which retrieved hit the citation
            references (with one-doc-per-hit layout, this maps
            directly to ``RagResult.citation.hits[document_index]``).
        document_title: Title of the cited document. For attune-rag
            this is the template path
            (e.g. ``concepts/tool-security-audit.md``).
        cited_text: Verbatim span the model attributes its claim to.
            Always rendered in the footnote so readers can verify
            without opening the source.
        cited_block_index: For ``custom_content`` documents, the
            block index within the document's content array. With
            attune-rag's one-block-per-document layout this is
            always ``0``; the field is kept for forward compat
            with multi-block layouts.
    """

    response_span: tuple[int, int]
    document_index: int
    document_title: str
    cited_text: str
    cited_block_index: int = 0


@dataclass(frozen=True)
class CitedSource:
    """A single cited source within a ``CitationRecord``.

    ``excerpt`` is optional — use it for a short quote or
    preview so humans skimming citations can verify
    relevance without opening every file.
    """

    template_path: str
    category: str
    score: float
    excerpt: str | None = None


@dataclass(frozen=True)
class CitationRecord:
    """Provenance for a single RAG pipeline run.

    Immutable so callers can safely include it in cached
    results, logs, or dataclass comparisons.
    """

    query: str
    hits: tuple[CitedSource, ...]
    retrieved_at: datetime
    retriever_name: str


def format_citations_markdown(
    record: CitationRecord,
    base_url: str | None = None,
) -> str:
    """Render a CitationRecord as a markdown section.

    If ``base_url`` is provided, template paths become
    clickable links. Empty citations render a clear
    "no grounding sources" message so consumers can
    distinguish "RAG ran but found nothing" from
    "RAG was never called."
    """
    if not record.hits:
        return "## Sources\n\nNo grounding sources available.\n"

    lines: list[str] = ["## Sources", ""]
    for source in record.hits:
        path = source.template_path
        if base_url:
            trimmed = base_url.rstrip("/")
            link = f"[{path}]({trimmed}/{path})"
        else:
            link = f"`{path}`"
        category = f" — {source.category}" if source.category else ""
        score = f" (score {source.score:.2f})"
        lines.append(f"- {link}{category}{score}")
        if source.excerpt:
            excerpt = source.excerpt.strip().replace("\n", " ")
            lines.append(f"  > {excerpt}")
    lines.append("")
    return "\n".join(lines)


def format_claim_citations_markdown(
    text: str,
    citations: Iterable[ClaimCitation],
    base_url: str | None = None,
) -> str:
    """Render response text with footnote-style claim citations.

    Each unique cited document becomes a footnote definition; an
    in-text marker like ``[^1]`` is inserted at the end of every
    cited span. Cited spans appear in the response in the order
    of their ``response_span`` start; identical document indexes
    share a footnote number so a reader sees "[^1]" twice when
    two claims cite the same source rather than ballooning the
    footnote list.

    The rendered markdown is *additive* — the original response
    text is preserved verbatim with markers spliced in at span
    ends. When citations is empty, a one-line italic note is
    prepended so consumers can distinguish "model didn't cite"
    from "this code path doesn't produce citations" (the legacy
    ``[P{n}]`` path returns text without calling this helper at
    all).

    Args:
        text: The assembled response text from ``CitedResponse``.
        citations: Iterable of ``ClaimCitation`` produced by the
            citations-capable provider.
        base_url: When set, footnote document titles render as
            clickable links (``[title](base_url/title)``).

    Returns:
        Markdown string. Stable, deterministic ordering for tests.
    """
    citations = list(citations)
    if not citations:
        return f"_No claim-level citations were emitted for this response._\n\n{text}\n"

    # Sort by span start so the in-text markers appear in narrative order.
    ordered = sorted(citations, key=lambda c: (c.response_span[0], c.response_span[1]))

    # Assign footnote numbers. Same document => same footnote number; the
    # first appearance order wins so a reader's eye flow matches the
    # footnote list at the bottom.
    footnote_num: dict[int, int] = {}
    for c in ordered:
        if c.document_index not in footnote_num:
            footnote_num[c.document_index] = len(footnote_num) + 1

    # Splice markers into the text at each citation's response_span end.
    # We process from rightmost to leftmost so earlier offsets stay valid.
    pieces: list[tuple[int, str]] = [
        (c.response_span[1], f"[^{footnote_num[c.document_index]}]") for c in ordered
    ]
    pieces.sort(key=lambda p: p[0], reverse=True)
    rendered = text
    for offset, marker in pieces:
        rendered = rendered[:offset] + marker + rendered[offset:]

    # Footnote definitions: one per unique document, in footnote-number
    # order. Each definition includes the cited_text so a reader can
    # verify without opening the source.
    footnote_lines: list[str] = []
    seen_docs: set[int] = set()
    for c in ordered:
        if c.document_index in seen_docs:
            continue
        seen_docs.add(c.document_index)
        n = footnote_num[c.document_index]
        title = c.document_title or "(untitled)"
        if base_url:
            trimmed = base_url.rstrip("/")
            label = f"[`{title}`]({trimmed}/{title})"
        else:
            label = f"`{title}`"
        excerpt = c.cited_text.strip().replace("\n", " ")
        footnote_lines.append(f'[^{n}]: {label} — "{excerpt}"')

    return rendered + "\n\n" + "\n".join(footnote_lines) + "\n"


def build_citation_record(
    query: str,
    hits: Iterable,  # Iterable[RetrievalHit] — typed loosely to avoid cycle
    retriever_name: str,
    retrieved_at: datetime,
    excerpt_chars: int = 200,
) -> CitationRecord:
    """Convert ``RetrievalHit`` objects into a ``CitationRecord``.

    Accepted as a helper here (not in pipeline.py) so any
    consumer can turn retrieval output into provenance
    without threading the pipeline through.
    """
    cited: list[CitedSource] = []
    for hit in hits:
        entry = hit.entry
        content = entry.content
        excerpt = content.strip()[:excerpt_chars] if content else None
        cited.append(
            CitedSource(
                template_path=entry.path,
                category=entry.category,
                score=hit.score,
                excerpt=excerpt,
            )
        )
    return CitationRecord(
        query=query,
        hits=tuple(cited),
        retrieved_at=retrieved_at,
        retriever_name=retriever_name,
    )
