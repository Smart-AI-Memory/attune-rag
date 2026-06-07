"""Hybrid retriever: keyword + embedding fused via Reciprocal Rank Fusion.

RRF combines two ranked lists without score calibration or tuning: each
document scores ``sum(1 / (rrf_k + rank))`` over the lists it appears in.
It is robust precisely because it ignores the (incomparable) raw scores
of token-overlap vs cosine similarity and uses only ranks.

Degrades gracefully: if the ``[embeddings]`` extra is unavailable, the
embedding leg yields nothing and retrieval falls back to keyword-only,
so a hybrid pipeline never hard-fails on a base install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .retrieval import KeywordRetriever, RetrievalHit

if TYPE_CHECKING:
    from .corpus.base import CorpusProtocol


class HybridRetriever:
    """Fuse a keyword retriever and an embedding retriever with RRF.

    Args:
        keyword: Lexical retriever. Defaults to :class:`KeywordRetriever`.
        embedding: Dense retriever. When ``None``, an
            :class:`~attune_rag.embedding.EmbeddingRetriever` is lazily
            constructed on first use; if the embeddings extra is missing
            at call time, the hybrid silently falls back to keyword-only.
        rrf_k: RRF constant (60 is the canonical default; larger flattens
            the rank weighting).
        candidate_pool: How many candidates to pull from each leg before
            fusing (over-fetch so a doc ranked low by one leg but high by
            the other can still surface). Floored at ``4 * k``.
        keyword_weight / embedding_weight: Per-leg multipliers on the RRF
            contribution. The default favors keyword (2:1) because on a
            keyword-*tuned* corpus (curated summaries/aliases/category
            weights) an equal blend trades away top-1 precision, while the
            recall@k generalization gain on unseen corpora is robust to
            the weighting. Raise ``keyword_weight`` further to fully
            protect a tuned corpus; lower it toward 1:1 to maximize the
            embedding contribution on unstructured/arbitrary corpora.
    """

    def __init__(
        self,
        keyword: Any = None,
        embedding: Any = None,
        rrf_k: int = 60,
        candidate_pool: int = 20,
        keyword_weight: float = 2.0,
        embedding_weight: float = 1.0,
    ) -> None:
        self.keyword = keyword or KeywordRetriever()
        self.embedding = embedding
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool
        self.keyword_weight = keyword_weight
        self.embedding_weight = embedding_weight
        self._embedding_disabled = False

    def _get_embedding(self) -> Any:
        if self.embedding is None and not self._embedding_disabled:
            from .embedding import EmbeddingRetriever

            self.embedding = EmbeddingRetriever()
        return self.embedding

    def retrieve(self, query: str, corpus: CorpusProtocol, k: int = 3):
        pool = max(self.candidate_pool, k * 4)
        kw_hits = list(self.keyword.retrieve(query, corpus, k=pool))

        emb = self._get_embedding()
        emb_hits: list[RetrievalHit] = []
        if emb is not None:
            try:
                emb_hits = list(emb.retrieve(query, corpus, k=pool))
            except RuntimeError:
                # Embeddings extra not installed at call time — disable the
                # leg for subsequent queries and fall back to keyword-only.
                self._embedding_disabled = True

        if not emb_hits:
            return kw_hits[:k]

        scores: dict[str, float] = {}
        entry_by_path: dict[str, Any] = {}
        reasons: dict[str, set[str]] = {}
        legs = (
            ("keyword", kw_hits, self.keyword_weight),
            ("embedding", emb_hits, self.embedding_weight),
        )
        for leg, hits, weight in legs:
            for rank, hit in enumerate(hits):
                path = hit.entry.path
                scores[path] = scores.get(path, 0.0) + weight / (self.rrf_k + rank + 1)
                entry_by_path[path] = hit.entry
                reasons.setdefault(path, set()).add(leg)

        fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [
            RetrievalHit(
                entry=entry_by_path[path],
                score=score,
                match_reason="rrf:" + "+".join(sorted(reasons[path])),
            )
            for path, score in fused
        ]
