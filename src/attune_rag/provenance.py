"""CitationRecord + provenance formatting. Implementation in task 1.5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class CitedSource:
    """A single cited source within a CitationRecord."""

    template_path: str
    category: str
    score: float
    excerpt: str | None = None


@dataclass(frozen=True)
class CitationRecord:
    """Provenance for a RAG run. Immutable by design."""

    query: str
    hits: tuple[CitedSource, ...]
    retrieved_at: datetime
    retriever_name: str


def format_citations_markdown(
    record: CitationRecord,
    base_url: str | None = None,
) -> str:
    """Render citations as markdown. Stub until task 1.5."""
    raise NotImplementedError(
        "format_citations_markdown is implemented in task 1.5 " "of the RAG grounding spec."
    )
