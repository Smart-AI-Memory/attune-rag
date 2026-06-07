"""M1 validation for docs/specs/confidence-gated-retrieval.

Re-measures the confidence-gated fusion finding on the expanded
>=30-query hard set (tests/golden/queries_corpus_b_hard.yaml) to check
the n=4 lift survives at n>=30 with ZERO attune-help regression.

Configs: keyword (default) | hybrid 2:1 (shipped) | embedding-only
(retrieval-32M) | gated fusion (T sweep). All torch-free, no API.

Usage:  PYTHONPATH=src python3 scripts/validate_gated_fusion.py
"""

from __future__ import annotations

from pathlib import Path

from attune_rag import RagPipeline
from attune_rag.benchmark import _load_queries
from attune_rag.corpus import DirectoryCorpus
from attune_rag.embedding import EmbeddingRetriever
from attune_rag.hybrid import HybridRetriever
from attune_rag.retrieval import KeywordRetriever

ROOT = Path(__file__).resolve().parent.parent / "tests" / "golden"
MODEL = "minishlab/potion-retrieval-32M"


class GatedRetriever:
    """Keyword-primary; embedding rescue when keyword top-1 < threshold."""

    def __init__(self, threshold: float, embedding):
        self.kw = KeywordRetriever(min_score=0.0)
        self.emb = embedding
        self.threshold = threshold

    def retrieve(self, query, corpus, k=3):
        kw_hits = self.kw.retrieve(query, corpus, k=k)
        if kw_hits and kw_hits[0].score >= self.threshold:
            return kw_hits
        return self.emb.retrieve(query, corpus, k=k)


def _metrics(queries, retriever, corpus=None):
    kw = {"retriever": retriever}
    if corpus is not None:
        kw["corpus"] = corpus
    pipe = RagPipeline(**kw)
    p = r = 0
    for q in queries:
        exp = set(q["expected_in_top_3"])
        hits = [h.template_path for h in pipe.run(q["query"], k=3).citation.hits]
        if hits and hits[0] in exp:
            p += 1
        if set(hits) & exp:
            r += 1
    n = len(queries) or 1
    return p / n, r / n


def main():
    cbh = _load_queries(ROOT / "queries_corpus_b_hard.yaml")
    hard = [q for q in cbh if q.get("difficulty") == "hard"]
    cb = DirectoryCorpus(ROOT / "corpus_b")
    help_q = _load_queries(ROOT / "queries.yaml")
    emb = EmbeddingRetriever(model_name=MODEL)

    print(f"hard n={len(hard)}  full-cbh n={len(cbh)}  attune-help n={len(help_q)}\n")
    hdr = f"{'config':28s} {'hardP@1':>8s}{'hardR@3':>8s} {'cbhP@1':>8s}{'cbhR@3':>8s} {'helpP@1':>8s}{'helpR@3':>8s}"
    print(hdr)
    print("-" * len(hdr))

    configs = [
        ("keyword (default)", lambda: KeywordRetriever()),
        ("hybrid 2:1 (8M, shipped)", lambda: HybridRetriever()),
        ("embedding-only (ret-32M)", lambda: GatedRetriever(1e9, emb)),  # always embed
    ]
    for t in (2.0, 3.0, 4.0, 5.0, 6.0):
        configs.append((f"gated T={t} (ret-32M)", lambda t=t: GatedRetriever(t, emb)))

    for label, factory in configs:
        hp, hr = _metrics(hard, factory(), cb)
        cp, cr = _metrics(cbh, factory(), cb)
        ap, ar = _metrics(help_q, factory())
        print(f"{label:28s} {hp:8.2f}{hr:8.2f} {cp:8.2f}{cr:8.2f} {ap:8.2f}{ar:8.2f}")

    print(
        "\nDecision gate (R1/R2/R3): gated must beat keyword on hard P@1 at "
        "n>=30 AND hold help P@1/R@3 = 1.00/1.00."
    )


if __name__ == "__main__":
    main()
