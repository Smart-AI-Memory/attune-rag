"""Unit tests for KeywordRetriever.

Uses a deterministic in-memory fake CorpusProtocol so tests
don't depend on the optional [attune-help] extra.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

import pytest

from attune_rag import KeywordRetriever, RetrievalEntry, RetrievalHit
from attune_rag.corpus.base import CorpusProtocol


class FakeCorpus(CorpusProtocol):
    """A minimal path-keyed corpus for testing."""

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


def _entry(
    path: str,
    content: str = "",
    summary: str | None = None,
    related: tuple[str, ...] = (),
    category: str = "concepts",
) -> RetrievalEntry:
    return RetrievalEntry(
        path=path,
        category=category,
        content=content,
        summary=summary,
        related=related,
    )


@pytest.fixture
def corpus() -> FakeCorpus:
    return FakeCorpus(
        [
            _entry(
                path="concepts/security-audit.md",
                summary="Run a security audit over your codebase",
                content="The security audit workflow scans for vulnerabilities.",
            ),
            _entry(
                path="concepts/smart-test.md",
                summary="Generate missing tests automatically",
                content="Smart test finds untested public APIs and writes tests.",
            ),
            _entry(
                path="concepts/code-review.md",
                summary="Review code quality and style",
                content="Code review flags style issues and potential bugs.",
            ),
            _entry(
                path="quickstarts/security-audit.md",
                summary="Quick start for security audit",
                content="Run 'attune workflow run security-audit' to get started.",
                related=("concepts/security-audit.md",),
            ),
            _entry(
                path="concepts/unrelated.md",
                summary="Something totally different",
                content="Haiku poetry under the moon, far from programming.",
            ),
        ]
    )


def test_exact_feature_token_match(corpus: FakeCorpus) -> None:
    hits = KeywordRetriever().retrieve("security audit", corpus)
    paths = [h.entry.path for h in hits]
    assert "concepts/security-audit.md" in paths
    assert "quickstarts/security-audit.md" in paths


def test_summary_only_match_wins_over_unrelated(corpus: FakeCorpus) -> None:
    hits = KeywordRetriever().retrieve("generate missing tests", corpus)
    top = hits[0]
    assert top.entry.path == "concepts/smart-test.md"


def test_below_threshold_returns_empty(corpus: FakeCorpus) -> None:
    hits = KeywordRetriever().retrieve("zzzzzz nonexistent xyzqq", corpus)
    assert hits == []


def test_respects_k(corpus: FakeCorpus) -> None:
    hits = KeywordRetriever().retrieve("security audit", corpus, k=1)
    assert len(hits) == 1


def test_deterministic_tie_breaking() -> None:
    # Two entries with identical score: tie break by path asc.
    # Query has two distinct tokens so each entry clears
    # MIN_SCORE via SUMMARY_WEIGHT * 2 = 3.0 >= 2.0.
    corpus = FakeCorpus(
        [
            _entry(path="b.md", summary="alpha beta gamma"),
            _entry(path="a.md", summary="alpha beta gamma"),
        ]
    )
    hits = KeywordRetriever().retrieve("alpha beta", corpus)
    assert [h.entry.path for h in hits] == ["a.md", "b.md"]


def test_empty_query_raises(corpus: FakeCorpus) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        KeywordRetriever().retrieve("", corpus)
    with pytest.raises(ValueError, match="non-empty"):
        KeywordRetriever().retrieve("   ", corpus)


def test_k_less_than_one_raises(corpus: FakeCorpus) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        KeywordRetriever().retrieve("security", corpus, k=0)


def test_match_reason_populated(corpus: FakeCorpus) -> None:
    hits = KeywordRetriever().retrieve("security audit", corpus)
    reasons = {h.match_reason for h in hits}
    assert all(r != "no-match" for r in reasons)


def test_related_entries_contribute_score() -> None:
    # Subclass lowers MIN_SCORE so RELATED_WEIGHT (0.5) can be
    # observed in isolation. In production the weight is a
    # bonus on top of path/summary/content hits.
    class LowThreshold(KeywordRetriever):
        MIN_SCORE = 0.1

    primary = _entry(
        path="a.md",
        summary="",
        content="body without the query tokens at all",
        related=("b.md",),
    )
    related = _entry(path="b.md", summary="alpha beta gamma")
    corpus = FakeCorpus([primary, related])
    hits = LowThreshold().retrieve("alpha beta", corpus)
    paths = {h.entry.path for h in hits}
    # b.md scores via its own summary; a.md scores via related-summary.
    assert paths == {"a.md", "b.md"}
    by_path = {h.entry.path: h for h in hits}
    assert "related" in by_path["a.md"].match_reason


def test_tunable_weights_affect_ranking() -> None:
    # Entry A scores via path; entry B scores via content.
    a = _entry(path="security-audit.md", content="body")
    b = _entry(path="other.md", content="security audit plus more security audit")
    corpus = FakeCorpus([a, b])

    class ContentHeavyRetriever(KeywordRetriever):
        PATH_WEIGHT = 0.1
        CONTENT_WEIGHT = 5.0
        MIN_SCORE = 0.0

    top = ContentHeavyRetriever().retrieve("security audit", corpus)[0]
    assert top.entry.path == "other.md"

    class PathHeavyRetriever(KeywordRetriever):
        PATH_WEIGHT = 10.0
        CONTENT_WEIGHT = 0.1
        MIN_SCORE = 0.0

    top = PathHeavyRetriever().retrieve("security audit", corpus)[0]
    assert top.entry.path == "security-audit.md"


def test_case_insensitive(corpus: FakeCorpus) -> None:
    lower = KeywordRetriever().retrieve("security audit", corpus)
    upper = KeywordRetriever().retrieve("SECURITY AUDIT", corpus)
    assert [h.entry.path for h in lower] == [h.entry.path for h in upper]


def test_punctuation_stripped(corpus: FakeCorpus) -> None:
    baseline = KeywordRetriever().retrieve("security audit", corpus)
    punct = KeywordRetriever().retrieve("security; audit!", corpus)
    assert [h.entry.path for h in baseline] == [h.entry.path for h in punct]


def test_retrievalhit_is_frozen(corpus: FakeCorpus) -> None:
    hit = KeywordRetriever().retrieve("security audit", corpus)[0]
    assert isinstance(hit, RetrievalHit)
    with pytest.raises(Exception):
        replace  # silences unused import warning
        hit.score = 99.0  # type: ignore[misc]
