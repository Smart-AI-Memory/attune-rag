"""Retriever protocol and default KeywordRetriever.

Implementation in task 1.4. Task 1.1 scaffolds the class
shapes so imports resolve.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .corpus.base import CorpusProtocol, RetrievalEntry


@dataclass(frozen=True)
class RetrievalHit:
    """A single retrieval result. Populated in task 1.4."""

    entry: RetrievalEntry
    score: float
    match_reason: str


@runtime_checkable
class RetrieverProtocol(Protocol):
    """Any object with a retrieve(query, corpus, k) method."""

    def retrieve(
        self,
        query: str,
        corpus: CorpusProtocol,
        k: int = 3,
    ) -> Iterable[RetrievalHit]: ...


class KeywordRetriever:
    """Keyword + tag based retriever. Implementation in task 1.4."""

    PATH_WEIGHT: float = 2.0
    SUMMARY_WEIGHT: float = 1.5
    CONTENT_WEIGHT: float = 1.0
    RELATED_WEIGHT: float = 0.5
    MIN_SCORE: float = 2.0

    def retrieve(
        self,
        query: str,
        corpus: CorpusProtocol,
        k: int = 3,
    ) -> list[RetrievalHit]:
        """Stub until task 1.4."""
        raise NotImplementedError(
            "KeywordRetriever.retrieve is implemented in " "task 1.4 of the RAG grounding spec."
        )
