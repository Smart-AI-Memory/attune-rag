"""Unit tests for KeywordRetriever.

Uses a deterministic in-memory fake CorpusProtocol so tests
don't depend on the optional [attune-help] extra.
"""

from __future__ import annotations

from collections.abc import Iterable

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
    from dataclasses import FrozenInstanceError

    hit = KeywordRetriever().retrieve("security audit", corpus)[0]
    assert isinstance(hit, RetrievalHit)
    with pytest.raises(FrozenInstanceError):
        hit.score = 99.0  # type: ignore[misc]


def test_category_weight_boosts_primary_over_lesson() -> None:
    # Same summary/content in two categories; concepts should
    # win because category weight 1.5 > 0.4.
    primary = _entry(
        path="concepts/tool-security-audit.md",
        category="concepts",
        summary="security audit",
    )
    lesson = _entry(
        path="errors/tool-security-audit-flag.md",
        category="errors",
        summary="security audit",
    )
    corpus = FakeCorpus([primary, lesson])
    hits = KeywordRetriever().retrieve("security audit", corpus)
    assert hits[0].entry.path == "concepts/tool-security-audit.md"


def test_category_weight_penalty_can_drop_below_min_score() -> None:
    # errors entry scores base 3.0 (2 summary hits × 1.5) * 0.4
    # = 1.2 < MIN_SCORE 2.0, so it is filtered entirely.
    lesson = _entry(
        path="errors/something.md",
        category="errors",
        summary="alpha beta",
    )
    corpus = FakeCorpus([lesson])
    hits = KeywordRetriever().retrieve("alpha beta", corpus)
    assert hits == []


def test_category_weight_reason_includes_category_marker() -> None:
    entry = _entry(
        path="concepts/x.md",
        category="concepts",
        summary="alpha beta",
    )
    corpus = FakeCorpus([entry])
    hits = KeywordRetriever().retrieve("alpha beta", corpus)
    assert hits and "cat:concepts" in hits[0].match_reason


def test_category_weight_reason_omitted_when_weight_is_one() -> None:
    entry = _entry(
        path="references/x.md",
        category="references",
        summary="alpha beta",
    )
    corpus = FakeCorpus([entry])
    hits = KeywordRetriever().retrieve("alpha beta", corpus)
    assert hits
    assert "cat:" not in hits[0].match_reason


def test_unknown_category_uses_default_weight() -> None:
    entry = _entry(
        path="nonstandard/x.md",
        category="nonstandard",
        summary="alpha beta gamma delta",
    )
    corpus = FakeCorpus([entry])
    hits = KeywordRetriever().retrieve("alpha beta", corpus)
    assert hits
    # Unknown category uses DEFAULT_CATEGORY_WEIGHT (1.0)
    # so the marker is omitted.
    assert "cat:" not in hits[0].match_reason


def test_path_hit_cap_limits_long_filenames() -> None:
    # A long error-style filename has 5 query-matching tokens
    # but should only contribute PATH_HIT_CAP (3) to the score.
    long_path = _entry(
        path="errors/alpha-beta-gamma-delta-epsilon.md",
        category="errors",
        summary="",
    )
    # Short feature-named concept with 1 summary hit only.
    short_concept = _entry(
        path="concepts/x.md",
        category="concepts",
        summary="alpha",
    )
    corpus = FakeCorpus([long_path, short_concept])

    class NoCategoryBias(KeywordRetriever):
        CATEGORY_WEIGHTS: dict[str, float] = {}
        PATH_HIT_CAP = 3

    class UncappedPath(KeywordRetriever):
        CATEGORY_WEIGHTS: dict[str, float] = {}
        PATH_HIT_CAP = 100

    capped = NoCategoryBias().retrieve("alpha beta gamma delta epsilon", corpus)
    uncapped = UncappedPath().retrieve("alpha beta gamma delta epsilon", corpus)

    # Without the cap, the long path dominates (5 × 2.0 = 10).
    assert uncapped[0].entry.path == "errors/alpha-beta-gamma-delta-epsilon.md"
    # With the cap (3 hits max), base path score is 6.0 — still
    # larger than short_concept's 1.5 but crucially doesn't scale
    # with filename length.
    top_capped_hits = capped[0].match_reason
    assert "path:3" in top_capped_hits


def test_stemming_matches_singular_and_plural() -> None:
    primary = _entry(
        path="concepts/bug.md",
        category="concepts",
        summary="catches bugs early",
    )
    corpus = FakeCorpus([primary])
    # "bugs" (query) stems to "bug"; "bug" matches.
    hits = KeywordRetriever().retrieve("catch bugs", corpus)
    assert hits and hits[0].entry.path == "concepts/bug.md"


def test_stemming_matches_ate_ator() -> None:
    primary = _entry(
        path="references/tool-doc-orchestrator.md",
        category="references",
        summary="orchestrator coordinates documentation workflows",
    )
    corpus = FakeCorpus([primary])
    # "orchestrate" stems to "orchestr"; "orchestrator" also
    # stems to "orchestr" via the "ator" suffix.
    hits = KeywordRetriever().retrieve("orchestrate documentation", corpus)
    assert hits and hits[0].entry.path == "references/tool-doc-orchestrator.md"


def test_stemming_preserves_short_tokens() -> None:
    # Tokens whose stemmed length would fall below
    # _MIN_STEM_LEN must be returned unchanged.
    from attune_rag.retrieval import _stem

    assert _stem("is") == "is"
    assert _stem("on") == "on"
    assert _stem("bug") == "bug"
