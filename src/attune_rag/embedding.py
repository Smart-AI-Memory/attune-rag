"""Embedding retriever using model2vec static embeddings.

Optional — requires the ``[embeddings]`` extra (``model2vec``, which
pulls ``numpy``). This module is import-safe without the extra: both
``numpy`` and ``model2vec`` are imported lazily inside methods, so
``from attune_rag.embedding import EmbeddingRetriever`` never fails on a
base install. The dependency only materializes when you actually embed.

Why static embeddings: model2vec distills a transformer into a static
lookup table, so it embeds with no torch, no GPU, fully offline, in
milliseconds — matching attune-rag's dependency-light, no-API-key
character while still capturing the paraphrase similarity that the
token-overlap :class:`~attune_rag.retrieval.KeywordRetriever` misses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .retrieval import RetrievalHit

if TYPE_CHECKING:
    from .corpus.base import CorpusProtocol, RetrievalEntry

#: Default distilled static model. ~30 MB, 256-dim, downloads once from
#: HuggingFace and is cached locally by model2vec.
DEFAULT_MODEL = "minishlab/potion-base-8M"


class EmbeddingRetriever:
    """Rank corpus entries by cosine similarity of static embeddings.

    The corpus embedding matrix is computed once per corpus object and
    cached (keyed by object identity), so a benchmark that reuses one
    pipeline across many queries pays the embedding cost only once.

    Args:
        model_name: model2vec model id. Ignored when ``encoder`` is set.
        encoder: An object exposing ``encode(list[str]) -> 2D array``.
            Injectable for tests so they never download a model. When
            ``None``, a model2vec ``StaticModel`` is lazily loaded.
        content_chars: Cap on body text included per entry's embedding
            input (static embeddings are order-insensitive; the head of
            the doc carries the topical signal without unbounded cost).
        query_prefix: Optional instruction prepended to the *query* (not
            the corpus entries) before encoding. Enables asymmetric
            retrieval models that expect a query instruction — e.g. BGE
            wants "Represent this sentence for searching relevant
            passages: ". Empty by default, so static/symmetric models
            (model2vec) are unaffected. See
            :class:`~attune_rag.transformer.TransformerRetriever`.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        encoder: Any = None,
        content_chars: int = 1000,
        query_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self._encoder = encoder
        self._content_chars = content_chars
        self._query_prefix = query_prefix
        self._cache: dict[int, tuple[list[Any], Any]] = {}

    def _get_encoder(self) -> Any:
        if (
            self._encoder is None
        ):  # pragma: no cover - real model load (I/O); tests inject a fake encoder
            try:
                from model2vec import StaticModel
            except ImportError as exc:
                raise RuntimeError(
                    "EmbeddingRetriever requires the embeddings extra. "
                    "Install with: pip install 'attune-rag[embeddings]'"
                ) from exc
            self._encoder = StaticModel.from_pretrained(self._model_name)
        return self._encoder

    def _entry_text(self, entry: RetrievalEntry) -> str:
        """Compose the text embedded for an entry: humanized title +
        summary (if any) + a budgeted slice of the body."""
        title = entry.path.rsplit("/", 1)[-1]
        title = title[:-3] if title.endswith(".md") else title
        title = title.replace("-", " ").replace("_", " ")
        parts = [title]
        if entry.summary:
            parts.append(entry.summary)
        if entry.content:
            parts.append(entry.content[: self._content_chars])
        return "\n".join(parts)

    def _corpus_matrix(self, corpus: CorpusProtocol) -> tuple[list[Any], Any]:
        cached = self._cache.get(id(corpus))
        if cached is not None:
            return cached
        import numpy as np

        entries = list(corpus.entries())
        if not entries:
            result: tuple[list[Any], Any] = (entries, np.zeros((0, 0)))
            self._cache[id(corpus)] = result
            return result
        mat = np.asarray(
            self._get_encoder().encode([self._entry_text(e) for e in entries]),
            dtype="float32",
        )
        mat = mat / _norms(mat)
        self._cache[id(corpus)] = (entries, mat)
        return entries, mat

    def retrieve(self, query: str, corpus: CorpusProtocol, k: int = 3):
        import numpy as np

        entries, mat = self._corpus_matrix(corpus)
        if not entries:
            return []
        q_text = self._query_prefix + query if self._query_prefix else query
        q = np.asarray(self._get_encoder().encode([q_text]), dtype="float32")
        q = q / _norms(q)
        sims = (mat @ q[0]).tolist()
        order = sorted(range(len(entries)), key=lambda i: (-sims[i], entries[i].path))[:k]
        return [
            RetrievalHit(entry=entries[i], score=float(sims[i]), match_reason="embedding")
            for i in order
        ]


def _norms(mat: Any) -> Any:
    """Row-wise L2 norms with zero rows clamped to 1.0 (avoid div-by-zero)."""
    import numpy as np

    n = np.linalg.norm(mat, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return n
