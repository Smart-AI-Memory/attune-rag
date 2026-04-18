"""Provenance records + markdown formatting."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime


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
