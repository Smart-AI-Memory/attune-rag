"""M2 mechanism bake-off for docs/specs/confidence-gated-retrieval.

Decides Q1 (hard switch vs below-gate RRF blend) and Q2 (gate signal:
keyword top-1 score vs top1-top2 gap), measured on:

- corpus_b hard set (26 hard + 6 medium — the R1 >=30-query set),
- corpus_c (20 hard + 4 medium — second arbitrary corpus, absent at M1),
- attune-help golden set (the zero-regression guard, R2).

All torch-free, deterministic, no API. The blend variant fuses
keyword+embedding at 1:1 below the gate (keyword is by definition
unconfident there, so the shipped 2:1 keyword bias is not applied).

Usage:  PYTHONPATH=src python3 scripts/measure_gated_mechanism.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import structlog

from attune_rag import RagPipeline
from attune_rag.benchmark import _load_queries
from attune_rag.corpus import DirectoryCorpus
from attune_rag.embedding import EmbeddingRetriever
from attune_rag.hybrid import HybridRetriever
from attune_rag.retrieval import KeywordRetriever

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

ROOT = Path(__file__).resolve().parent.parent / "tests" / "golden"
MODEL = "minishlab/potion-retrieval-32M"


class GatedRetriever:
    """Keyword-primary retrieval with a confidence gate.

    Above the gate the keyword hits are returned untouched. Below it,
    ``mechanism`` decides the rescue: ``switch`` goes embedding-only,
    ``blend`` RRF-fuses keyword+embedding at 1:1. ``signal`` keys the
    gate on the keyword top-1 score or the top1-top2 gap.
    """

    def __init__(self, threshold, embedding, *, signal="score", mechanism="switch"):
        self.kw = KeywordRetriever(min_score=0.0)
        self.emb = embedding
        self.threshold = threshold
        self.signal = signal
        self.mechanism = mechanism
        self._blend = HybridRetriever(
            keyword=self.kw,
            embedding=embedding,
            keyword_weight=1.0,
            embedding_weight=1.0,
        )

    def _confidence(self, hits) -> float:
        if not hits:
            return 0.0
        if self.signal == "score":
            return hits[0].score
        top2 = hits[1].score if len(hits) > 1 else 0.0
        return hits[0].score - top2

    def retrieve(self, query, corpus, k=3):
        kw_hits = self.kw.retrieve(query, corpus, k=k)
        if self._confidence(kw_hits) >= self.threshold:
            return kw_hits
        if self.mechanism == "switch":
            return self.emb.retrieve(query, corpus, k=k)
        return self._blend.retrieve(query, corpus, k=k)


def _metrics(queries, retriever, corpus=None):
    kw = {"retriever": retriever}
    if corpus is not None:
        kw["corpus"] = corpus
    pipe = RagPipeline(**kw)
    p = r = 0
    for q in queries:
        exp = set(q["expected_in_top_3"])
        hits = [h.template_path for h in pipe.run(q["query"], k=3).citation.hits]
        p += bool(hits and hits[0] in exp)
        r += bool(set(hits) & exp)
    n = len(queries) or 1
    return p / n, r / n


def main():
    cb = DirectoryCorpus(ROOT / "corpus_b")
    cc = DirectoryCorpus(ROOT / "corpus_c")
    cbh = _load_queries(ROOT / "queries_corpus_b_hard.yaml")
    cch = _load_queries(ROOT / "queries_corpus_c_hard.yaml")
    help_q = _load_queries(ROOT / "queries.yaml")

    tiers = {
        "cb-hard": ([q for q in cbh if q.get("difficulty") == "hard"], cb),
        "cb-med": ([q for q in cbh if q.get("difficulty") == "medium"], cb),
        "cc-hard": ([q for q in cch if q.get("difficulty") == "hard"], cc),
        "cc-med": ([q for q in cch if q.get("difficulty") == "medium"], cc),
        "help": (help_q, None),
    }
    for name, (qs, _) in tiers.items():
        print(f"{name}: n={len(qs)}", end="  ")
    print("\n")

    emb = EmbeddingRetriever(model_name=MODEL)

    configs = [
        ("keyword (default)", lambda: KeywordRetriever()),
        ("hybrid 2:1 (8M, shipped)", lambda: HybridRetriever()),
        ("embedding-only (ret-32M)", lambda: GatedRetriever(1e9, emb)),
    ]
    for t in (2.0, 3.0, 4.0, 5.0, 6.0):
        configs.append((f"switch/score T={t:g}", lambda t=t: GatedRetriever(t, emb)))
    for t in (2.0, 3.0, 4.0, 5.0, 6.0):
        configs.append(
            (
                f"blend/score T={t:g}",
                lambda t=t: GatedRetriever(t, emb, mechanism="blend"),
            )
        )
    for g in (1.0, 2.0, 3.0, 4.0):
        configs.append((f"switch/gap G={g:g}", lambda g=g: GatedRetriever(g, emb, signal="gap")))
    for g in (1.0, 2.0, 3.0, 4.0):
        configs.append(
            (
                f"blend/gap G={g:g}",
                lambda g=g: GatedRetriever(g, emb, signal="gap", mechanism="blend"),
            )
        )

    cols = list(tiers)
    hdr = f"{'config':26s}" + "".join(f" {c:>13s}" for c in cols)
    print(hdr)
    print("-" * len(hdr))
    for label, factory in configs:
        retr = factory()
        cells = []
        for name in cols:
            qs, corpus = tiers[name]
            p, r = _metrics(qs, retr, corpus)
            cells.append(f"{p:.2f}/{r:.2f}")
        print(f"{label:26s}" + "".join(f" {c:>13s}" for c in cells))

    print(
        "\nQ1/Q2 gate: pick the mechanism+signal that maximizes hard-tier"
        "\nlift on BOTH corpora subject to help staying 1.00/1.00 and the"
        "\nmedium tier not regressing vs switch/score (the M1 shape)."
    )


if __name__ == "__main__":
    main()
