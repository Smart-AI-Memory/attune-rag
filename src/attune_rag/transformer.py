"""TransformerRetriever — opt-in dense retrieval with a real transformer.

Optional — requires the ``[transformers]`` extra (``sentence-transformers``,
which pulls ``torch``). Import-safe without the extra: ``sentence_transformers``
is imported lazily inside :meth:`_get_encoder`, so
``from attune_rag.transformer import TransformerRetriever`` never fails on a
base install. The (heavy) dependency only materializes when you actually embed.

When to reach for this vs the lighter retrievers
------------------------------------------------
This is the **heaviest** rung of the opt-in ladder
(``keyword`` → ``[embeddings]`` static hybrid → ``[transformers]``). Use it
only for an **arbitrary corpus where paraphrase recall matters** and the
torch-free options fall short:

- Bundled / keyword-tuned corpus → ``KeywordRetriever`` (already optimal).
- Arbitrary corpus, lexically-aligned queries → ``[embeddings]``
  ``HybridRetriever`` (cheap, torch-free).
- Arbitrary corpus, heavy paraphrasing → **this**.

It is **embedding-primary** (a transformer leg, not gated behind keyword)
and tanks a keyword-tuned corpus's top-1 precision, which is exactly why it
is opt-in and never a default. On two unseen corpora it lifted hard-tier
paraphrase precision@1 well past every torch-free option (≈0.50 → 0.85–0.90)
— at the cost of a ~GB torch dependency and ~10–300 ms/query.

Cost: torch (~GB) + a one-time model download (then offline/deterministic),
~3 s first-load, ~10–300 ms/query (vs <1 ms keyword / ~1 ms static).
"""

from __future__ import annotations

from typing import Any

from .embedding import EmbeddingRetriever

#: Default retrieval-tuned transformer. ~130 MB, 384-dim. Best measured
#: hard-tier precision@1 of the candidates; downloads once from HuggingFace
#: and is cached locally by sentence-transformers.
DEFAULT_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"

#: BGE-v1.5 models are trained for asymmetric retrieval: the *query* gets an
#: instruction prefix while passages do not. Prepending it measurably lifts
#: precision (≈+5 pts hard-tier in validation). Pass ``query_prefix=""`` for
#: symmetric models (e.g. all-MiniLM-L6-v2).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class TransformerRetriever(EmbeddingRetriever):
    """Dense retriever backed by a sentence-transformers model.

    Reuses all of :class:`~attune_rag.embedding.EmbeddingRetriever`
    (per-corpus matrix cache, cosine ranking, the ``query_prefix``
    asymmetric hook); only the encoder differs — a lazily-loaded
    ``SentenceTransformer`` instead of a model2vec ``StaticModel``.

    Args:
        model_name: sentence-transformers model id. Defaults to
            ``BAAI/bge-small-en-v1.5``. Ignored when ``encoder`` is set.
        encoder: An object exposing ``encode(list[str]) -> 2D array``
            (the ``SentenceTransformer.encode`` interface). Injectable so
            tests never download torch or a model.
        content_chars: Body-text cap per entry (see base class).
        query_prefix: Instruction prepended to the query. Defaults to
            :data:`BGE_QUERY_PREFIX` (correct for the default BGE model).
            Set to ``""`` for symmetric models.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TRANSFORMER_MODEL,
        encoder: Any = None,
        content_chars: int = 1000,
        query_prefix: str = BGE_QUERY_PREFIX,
    ) -> None:
        super().__init__(
            model_name=model_name,
            encoder=encoder,
            content_chars=content_chars,
            query_prefix=query_prefix,
        )

    def _get_encoder(self) -> Any:
        if (
            self._encoder is None
        ):  # pragma: no cover - real model load (I/O); tests inject a fake encoder
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "TransformerRetriever requires the [transformers] extra. "
                    "Install with: pip install 'attune-rag[transformers]'"
                ) from exc
            self._encoder = SentenceTransformer(self._model_name)
        return self._encoder
