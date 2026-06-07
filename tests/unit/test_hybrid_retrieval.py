"""Tests for Phase 3 hybrid + embedding retrieval.

The RRF fusion / weighting / fallback logic is pure-Python and tested
with stub retrievers (no model). The embedding cosine path is tested
with an injected fake encoder (no model download); it needs numpy, so
those tests importorskip it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from attune_rag.hybrid import HybridRetriever
from attune_rag.retrieval import RetrievalHit


def _entry(path: str) -> SimpleNamespace:
    return SimpleNamespace(path=path, summary=None, content="", category="")


class _StubRetriever:
    """Returns a fixed ranked list of (path, score) as RetrievalHits."""

    def __init__(self, ranked: list[tuple[str, float]], raises: bool = False) -> None:
        self._ranked = ranked
        self._raises = raises

    def retrieve(self, query: str, corpus, k: int = 3):
        if self._raises:
            raise RuntimeError("embeddings extra not installed")
        return [
            RetrievalHit(entry=_entry(p), score=s, match_reason="stub") for p, s in self._ranked[:k]
        ]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------


def test_rrf_ranks_docs_in_both_legs_above_single_leg() -> None:
    kw = _StubRetriever([("A", 9), ("B", 8), ("C", 7)])
    emb = _StubRetriever([("C", 0.9), ("D", 0.8), ("A", 0.7)])
    h = HybridRetriever(keyword=kw, embedding=emb, keyword_weight=1.0, embedding_weight=1.0)
    paths = [hit.entry.path for hit in h.retrieve("q", corpus=None, k=4)]
    # A and C appear in BOTH legs -> they should outrank B and D (single-leg).
    assert set(paths[:2]) == {"A", "C"}
    assert paths.index("A") < paths.index("B")
    assert paths.index("C") < paths.index("D")


def test_match_reason_records_fused_legs() -> None:
    kw = _StubRetriever([("A", 9)])
    emb = _StubRetriever([("A", 0.9)])
    h = HybridRetriever(keyword=kw, embedding=emb)
    hit = list(h.retrieve("q", corpus=None, k=1))[0]
    assert hit.match_reason == "rrf:embedding+keyword"


def test_keyword_weight_protects_keyword_top1() -> None:
    # keyword and embedding disagree on the #1. With a high keyword weight,
    # keyword's top-1 should win the fused ranking.
    kw = _StubRetriever([("KW", 9), ("Z", 1)])
    emb = _StubRetriever([("EMB", 0.9), ("W", 0.1)])  # disjoint from keyword
    h = HybridRetriever(keyword=kw, embedding=emb, keyword_weight=5.0, embedding_weight=1.0)
    top = [hit.entry.path for hit in h.retrieve("q", corpus=None, k=2)]
    assert top[0] == "KW"  # high keyword weight keeps keyword's top-1 on top


# ---------------------------------------------------------------------------
# Graceful fallback to keyword-only
# ---------------------------------------------------------------------------


def test_hybrid_lazily_constructs_embedding_retriever() -> None:
    # No embedding injected -> _get_embedding builds a default
    # EmbeddingRetriever (construction is lazy: no model load, no numpy).
    from attune_rag.embedding import EmbeddingRetriever

    h = HybridRetriever()
    assert isinstance(h._get_embedding(), EmbeddingRetriever)


def test_hybrid_keyword_only_when_embedding_disabled() -> None:
    h = HybridRetriever(keyword=_StubRetriever([("A", 9), ("B", 8)]))
    h._embedding_disabled = True  # _get_embedding -> None -> embedding leg skipped
    paths = [hit.entry.path for hit in h.retrieve("q", corpus=None, k=2)]
    assert paths == ["A", "B"]


def test_falls_back_to_keyword_when_embedding_empty() -> None:
    kw = _StubRetriever([("A", 9), ("B", 8)])
    emb = _StubRetriever([])  # no embedding hits
    h = HybridRetriever(keyword=kw, embedding=emb)
    paths = [hit.entry.path for hit in h.retrieve("q", corpus=None, k=2)]
    assert paths == ["A", "B"]


def test_falls_back_and_disables_when_extra_missing() -> None:
    kw = _StubRetriever([("A", 9), ("B", 8)])
    emb = _StubRetriever([("Z", 0.9)], raises=True)  # simulates missing extra
    h = HybridRetriever(keyword=kw, embedding=emb)
    paths = [hit.entry.path for hit in h.retrieve("q", corpus=None, k=2)]
    assert paths == ["A", "B"]  # keyword-only
    assert h._embedding_disabled is True
    # second call must not retry the embedding leg
    paths2 = [hit.entry.path for hit in h.retrieve("q", corpus=None, k=2)]
    assert paths2 == ["A", "B"]


# ---------------------------------------------------------------------------
# EmbeddingRetriever (fake encoder, no model download)
# ---------------------------------------------------------------------------


def _fake_corpus(entries):
    return SimpleNamespace(entries=lambda: list(entries))


def test_embedding_entry_text_composition() -> None:
    from attune_rag.embedding import EmbeddingRetriever

    r = EmbeddingRetriever(encoder=object())  # encoder unused by _entry_text
    full = r._entry_text(
        SimpleNamespace(
            path="howto/define-a-job.md", summary="how to register a job", content="body text"
        )
    )
    assert "define a job" in full  # .md stripped, hyphens humanized
    assert "how to register a job" in full  # summary included
    assert "body text" in full  # content included

    minimal = r._entry_text(SimpleNamespace(path="x/no_meta.md", summary=None, content=""))
    assert minimal == "no meta"  # title only when no summary/content


def test_embedding_retriever_ranks_by_cosine() -> None:
    np = pytest.importorskip("numpy")
    from attune_rag.embedding import EmbeddingRetriever

    vecs = {
        "match this job": np.array([1.0, 0.0]),
        "unrelated topic": np.array([0.0, 1.0]),
        "job query": np.array([0.9, 0.1]),  # closest to "match this job"
    }

    class FakeEncoder:
        def encode(self, texts):
            return np.array([vecs[t] for t in texts])

    entries = [
        SimpleNamespace(path="a.md", summary=None, content="match this job"),
        SimpleNamespace(path="b.md", summary=None, content="unrelated topic"),
    ]
    # _entry_text composes "title\ncontent"; make the title match the vec key
    for e in entries:
        e.path = e.content.replace(" ", "-") + ".md"
    # simplest: stub _entry_text by mapping content directly
    r = EmbeddingRetriever(encoder=FakeEncoder())
    r._entry_text = lambda e: e.content  # type: ignore[method-assign]
    hits = list(r.retrieve("job query", _fake_corpus(entries), k=1))
    assert hits[0].entry.content == "match this job"
    assert hits[0].match_reason == "embedding"


def test_embedding_retriever_empty_corpus() -> None:
    pytest.importorskip("numpy")
    from attune_rag.embedding import EmbeddingRetriever

    class FakeEncoder:
        def encode(self, texts):  # pragma: no cover - not reached on empty corpus
            raise AssertionError("should not encode an empty corpus")

    r = EmbeddingRetriever(encoder=FakeEncoder())
    assert list(r.retrieve("q", _fake_corpus([]), k=3)) == []


def test_embedding_retriever_caches_corpus_matrix() -> None:
    np = pytest.importorskip("numpy")
    from attune_rag.embedding import EmbeddingRetriever

    calls = {"n": 0}

    class CountingEncoder:
        def encode(self, texts):
            calls["n"] += 1
            return np.array([[float(len(t)), 1.0] for t in texts])

    entries = [
        SimpleNamespace(path="a.md", summary=None, content="alpha"),
        SimpleNamespace(path="b.md", summary=None, content="beta"),
    ]
    corpus = _fake_corpus(entries)
    r = EmbeddingRetriever(encoder=CountingEncoder())
    r.retrieve("x", corpus, k=1)
    r.retrieve("y", corpus, k=1)
    # 1 encode for the corpus matrix (cached) + 1 per query = 3, not 4.
    assert calls["n"] == 3
