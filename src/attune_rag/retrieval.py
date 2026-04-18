"""Keyword retriever and retriever protocol."""

from __future__ import annotations

import re
import string
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .corpus.base import CorpusProtocol, RetrievalEntry


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "how",
        "do",
        "does",
        "i",
        "to",
        "with",
        "for",
        "is",
        "are",
        "of",
        "in",
        "on",
        "at",
        "and",
        "or",
        "but",
        "can",
        "should",
        "would",
        "will",
        "be",
        "been",
        "by",
        "my",
        "me",
        "we",
        "it",
        "this",
        "that",
        "these",
        "those",
    }
)


def _tokenize(text: str) -> set[str]:
    lowered = _PUNCT_RE.sub(" ", text.lower())
    return {tok for tok in lowered.split() if tok and tok not in _STOPWORDS}


@dataclass(frozen=True)
class RetrievalHit:
    """A single retrieval result."""

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
    """Token-overlap retriever with path / summary / content / related weights.

    Weights and the minimum score are class attributes so the
    benchmark (task 2.4) can sweep parameters without editing
    source.
    """

    PATH_WEIGHT: float = 2.0
    SUMMARY_WEIGHT: float = 1.5
    CONTENT_WEIGHT: float = 1.0
    RELATED_WEIGHT: float = 0.5
    MIN_SCORE: float = 2.0
    CONTENT_PREVIEW_CHARS: int = 500

    def _score_entry(
        self,
        query_tokens: set[str],
        entry: RetrievalEntry,
        corpus: CorpusProtocol,
    ) -> tuple[float, str]:
        path_tokens = _tokenize(entry.path)
        summary_tokens = _tokenize(entry.summary or "")
        content_preview = entry.content[: self.CONTENT_PREVIEW_CHARS]
        content_tokens = _tokenize(content_preview)

        related_summary_tokens: set[str] = set()
        for related_path in entry.related:
            related_entry = corpus.get(related_path)
            if related_entry is None or not related_entry.summary:
                continue
            related_summary_tokens |= _tokenize(related_entry.summary)

        path_hits = len(query_tokens & path_tokens)
        summary_hits = len(query_tokens & summary_tokens)
        content_hits = len(query_tokens & content_tokens)
        related_hits = len(query_tokens & related_summary_tokens)

        score = (
            self.PATH_WEIGHT * path_hits
            + self.SUMMARY_WEIGHT * summary_hits
            + self.CONTENT_WEIGHT * content_hits
            + self.RELATED_WEIGHT * related_hits
        )

        reasons: list[str] = []
        if path_hits:
            reasons.append(f"path:{path_hits}")
        if summary_hits:
            reasons.append(f"summary:{summary_hits}")
        if content_hits:
            reasons.append(f"content:{content_hits}")
        if related_hits:
            reasons.append(f"related:{related_hits}")
        match_reason = "+".join(reasons) if reasons else "no-match"

        return score, match_reason

    def retrieve(
        self,
        query: str,
        corpus: CorpusProtocol,
        k: int = 3,
    ) -> list[RetrievalHit]:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k < 1:
            raise ValueError("k must be >= 1")

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored: list[RetrievalHit] = []
        for entry in corpus.entries():
            score, reason = self._score_entry(query_tokens, entry, corpus)
            if score >= self.MIN_SCORE:
                scored.append(RetrievalHit(entry=entry, score=score, match_reason=reason))

        scored.sort(key=lambda h: (-h.score, h.entry.path))
        return scored[:k]
