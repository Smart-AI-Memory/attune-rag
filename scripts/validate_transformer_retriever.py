"""M1 validation for docs/specs/transformer-retriever.

Reproduces the transformer-vs-torch-free margin on a SECOND arbitrary
corpus (corpus_c, an HTTP API client doc set — different domain/jargon
from corpus_b) to confirm the advantage generalizes before building.

Requires the [transformers] + [embeddings] extras and downloads models.
NOT run in CI (CI uses fake encoders). Manual reproduction:

    PYTHONPATH=src python3 scripts/validate_transformer_retriever.py
"""

from __future__ import annotations

from pathlib import Path

from attune_rag import RagPipeline, TransformerRetriever
from attune_rag.benchmark import _load_queries
from attune_rag.corpus import DirectoryCorpus
from attune_rag.embedding import EmbeddingRetriever
from attune_rag.retrieval import KeywordRetriever

ROOT = Path(__file__).resolve().parent.parent / "tests" / "golden"


def _metrics(queries, retriever, corpus):
    pipe = RagPipeline(corpus=corpus, retriever=retriever)
    p = r = 0
    for q in queries:
        exp = set(q["expected_in_top_3"])
        hits = [h.template_path for h in pipe.run(q["query"], k=3).citation.hits]
        p += bool(hits and hits[0] in exp)
        r += bool(set(hits) & exp)
    n = len(queries) or 1
    return p / n, r / n


def main():
    cc = DirectoryCorpus(ROOT / "corpus_c")
    queries = _load_queries(ROOT / "queries_corpus_c_hard.yaml")
    hard = [q for q in queries if q.get("difficulty") == "hard"]
    print(f"corpus_c (HTTP API client) hard n={len(hard)} / full n={len(queries)}\n")

    rows = [
        ("keyword", KeywordRetriever()),
        ("static (model2vec potion-8M)", EmbeddingRetriever()),
        ("transformer (bge-small, default)", TransformerRetriever()),
        ("transformer (symmetric)", TransformerRetriever(query_prefix="")),
    ]
    for label, retr in rows:
        hp, hr = _metrics(hard, retr, cc)
        fp, fr = _metrics(queries, retr, cc)
        print(f"{label:34s} hard P@1={hp:.2f} R@3={hr:.2f} | full P@1={fp:.2f} R@3={fr:.2f}")

    print(
        "\nR1 gate: transformer must beat the torch-free ceiling on a 2nd "
        "corpus. Validated 2026-06-07: keyword 0.25 / static 0.55 / "
        "transformer 0.90 hard P@1."
    )


if __name__ == "__main__":
    main()
