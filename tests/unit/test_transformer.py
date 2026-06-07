"""Tests for the opt-in TransformerRetriever + EmbeddingRetriever query_prefix.

All tests inject a fake encoder, so they never import torch /
sentence-transformers or download a model — the [transformers] extra is
not needed to run them. numpy (from the embeddings extra) is required for
the cosine path, so these importorskip it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from attune_rag.transformer import (
    BGE_QUERY_PREFIX,
    DEFAULT_TRANSFORMER_MODEL,
    TransformerRetriever,
)


def _fake_corpus(entries):
    return SimpleNamespace(entries=lambda: list(entries))


def test_query_prefix_applied_to_query_only() -> None:
    """EmbeddingRetriever.query_prefix prepends to the query, never the corpus."""
    np = pytest.importorskip("numpy")
    from attune_rag.embedding import EmbeddingRetriever

    seen: list[list[str]] = []

    class RecordingEncoder:
        def encode(self, texts):
            seen.append(list(texts))
            return np.array([[float(len(t)), 1.0] for t in texts])

    entries = [SimpleNamespace(path="a.md", summary=None, content="alpha")]
    r = EmbeddingRetriever(encoder=RecordingEncoder(), query_prefix="PFX: ")
    r._entry_text = lambda e: e.content  # type: ignore[method-assign]
    r.retrieve("hello", _fake_corpus(entries), k=1)

    # First encode call = corpus (no prefix); second = query (prefixed).
    assert seen[0] == ["alpha"]
    assert seen[1] == ["PFX: hello"]


def test_empty_query_prefix_is_noop() -> None:
    np = pytest.importorskip("numpy")
    from attune_rag.embedding import EmbeddingRetriever

    seen: list[list[str]] = []

    class RecordingEncoder:
        def encode(self, texts):
            seen.append(list(texts))
            return np.array([[float(len(t)), 1.0] for t in texts])

    entries = [SimpleNamespace(path="a.md", summary=None, content="alpha")]
    r = EmbeddingRetriever(encoder=RecordingEncoder())  # default query_prefix=""
    r._entry_text = lambda e: e.content  # type: ignore[method-assign]
    r.retrieve("hello", _fake_corpus(entries), k=1)

    assert seen[1] == ["hello"]  # unprefixed


def test_transformer_retriever_defaults() -> None:
    r = TransformerRetriever(encoder=object())
    assert r._model_name == DEFAULT_TRANSFORMER_MODEL
    assert r._query_prefix == BGE_QUERY_PREFIX


def test_transformer_retriever_ranks_by_cosine() -> None:
    """End-to-end retrieve() with an injected encoder — no torch."""
    np = pytest.importorskip("numpy")

    vecs = {
        # query (with BGE prefix) lands closest to the "auth" doc
        BGE_QUERY_PREFIX + "how do I log in": np.array([1.0, 0.0]),
        "authentication": np.array([0.95, 0.05]),
        "pagination": np.array([0.0, 1.0]),
    }

    class FakeEncoder:
        def encode(self, texts):
            return np.array([vecs[t] for t in texts])

    entries = [
        SimpleNamespace(path="authentication.md", summary=None, content="authentication"),
        SimpleNamespace(path="pagination.md", summary=None, content="pagination"),
    ]
    r = TransformerRetriever(encoder=FakeEncoder())
    r._entry_text = lambda e: e.content  # type: ignore[method-assign]
    hits = list(r.retrieve("how do I log in", _fake_corpus(entries), k=1))
    assert hits[0].entry.path == "authentication.md"
    assert hits[0].match_reason == "embedding"


def test_transformer_retriever_symmetric_override() -> None:
    """query_prefix='' supports symmetric models (e.g. MiniLM)."""
    r = TransformerRetriever(encoder=object(), query_prefix="")
    assert r._query_prefix == ""


def test_transformer_retriever_missing_extra_raises() -> None:
    """Without an injected encoder and without the extra, a helpful error."""
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("no sentence_transformers")
        return real_import(name, *args, **kwargs)

    r = TransformerRetriever()  # encoder=None -> will try to lazy-load
    builtins.__import__ = blocked_import
    try:
        with pytest.raises(RuntimeError, match=r"\[transformers\] extra"):
            r._get_encoder()
    finally:
        builtins.__import__ = real_import
