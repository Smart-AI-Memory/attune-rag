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


_STEM_SUFFIXES: tuple[str, ...] = (
    "ations",
    "ation",
    "ators",
    "ator",
    "ates",
    "ate",
    "ings",
    "ing",
    "ions",
    "ion",
    "ies",
    "ers",
    "ed",
    "er",
    "es",
    "s",
)
_MIN_STEM_LEN: int = 3


def _stem(token: str) -> str:
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _MIN_STEM_LEN:
            return token[: -len(suffix)]
    return token


def _tokenize(text: str) -> set[str]:
    lowered = _PUNCT_RE.sub(" ", text.lower())
    return {_stem(tok) for tok in lowered.split() if tok and tok not in _STOPWORDS}


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


_DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    # Primary, user-facing explanations — boost.
    "concepts": 1.5,
    "quickstarts": 1.5,
    "tasks": 1.5,
    # Reference and comparison material — neutral.
    "references": 1.0,
    "comparisons": 1.0,
    # Incidental guidance — mild penalty.
    "tips": 0.9,
    "notes": 0.9,
    "troubleshooting": 0.9,
    # Lesson-style content that often collides with primary
    # material on keyword overlap — strong penalty. These
    # categories tend to have long, keyword-dense filenames
    # (``errors/bug-predict-dangerous-eval-flags-*.md``) which
    # inflate path-hit scores versus the short feature-named
    # concept file they document.
    "errors": 0.4,
    "warnings": 0.4,
    "faqs": 0.4,
}


class KeywordRetriever:
    """Token-overlap retriever with path / summary / content / related weights.

    Weights and the minimum score are class attributes so the
    benchmark (task 2.4) can sweep parameters without editing
    source.

    ``CATEGORY_WEIGHTS`` applies a multiplier based on
    ``entry.category`` (the top-level directory in DirectoryCorpus /
    AttuneHelpCorpus). Tunes retrieval to prefer primary
    documentation (``concepts/``, ``quickstarts/``) over lesson-
    style material (``errors/``, ``warnings/``, ``faqs/``) when the
    two compete on keyword overlap. Unlisted categories get
    ``DEFAULT_CATEGORY_WEIGHT``.
    """

    PATH_WEIGHT: float = 2.0
    SUMMARY_WEIGHT: float = 1.5
    CONTENT_WEIGHT: float = 1.0
    RELATED_WEIGHT: float = 0.5
    MIN_SCORE: float = 2.0
    CONTENT_PREVIEW_CHARS: int = 500
    CATEGORY_WEIGHTS: dict[str, float] = _DEFAULT_CATEGORY_WEIGHTS
    DEFAULT_CATEGORY_WEIGHT: float = 1.0
    # Path hits are capped so long keyword-dense filenames
    # (common in ``errors/`` and ``warnings/``) don't dominate
    # ranking. The first ``PATH_HIT_CAP`` overlaps cover the
    # typical feature-name length; beyond that, path tokens are
    # usually lesson-title noise rather than relevance signal.
    PATH_HIT_CAP: int = 3

    def _category_weight(self, entry: RetrievalEntry) -> float:
        return self.CATEGORY_WEIGHTS.get(entry.category, self.DEFAULT_CATEGORY_WEIGHT)

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

        path_hits_raw = len(query_tokens & path_tokens)
        path_hits = min(path_hits_raw, self.PATH_HIT_CAP)
        summary_hits = len(query_tokens & summary_tokens)
        content_hits = len(query_tokens & content_tokens)
        related_hits = len(query_tokens & related_summary_tokens)

        base_score = (
            self.PATH_WEIGHT * path_hits
            + self.SUMMARY_WEIGHT * summary_hits
            + self.CONTENT_WEIGHT * content_hits
            + self.RELATED_WEIGHT * related_hits
        )
        category_weight = self._category_weight(entry)
        score = base_score * category_weight

        reasons: list[str] = []
        if path_hits:
            reasons.append(f"path:{path_hits}")
        if summary_hits:
            reasons.append(f"summary:{summary_hits}")
        if content_hits:
            reasons.append(f"content:{content_hits}")
        if related_hits:
            reasons.append(f"related:{related_hits}")
        if category_weight != 1.0 and base_score > 0:
            reasons.append(f"cat:{entry.category}×{category_weight:g}")
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
