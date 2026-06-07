# Tasks — RAG Strengthening

## Phase 1 — Harden the eval

**Status**: complete (2026-06-06) — advisory measurement, no gate changes.

- [x] **1.1** Out-of-corpus negative set (`tests/golden/queries_negative.yaml`, 12 queries: far + near).
- [x] **1.2** `_run_negative_benchmark` → `false_answer_rate` / `abstention_rate` + per-query leak detail.
- [x] **1.3** Promote per-difficulty rollup into the report dict (`_aggregate_by_difficulty`); JSON-visible, not just printed.
- [x] **1.4** Extended advisory hard set (`tests/golden/queries_extended.yaml`, 5 hard queries) measured via `--extended`. Kept OUT of the gated set so `queries.yaml`'s `queries_sha256` lock holds.
- [x] **1.5** Wire `--negatives` / `--extended` into `benchmark.py`; include in JSON dumps. Both advisory — never gate.
- [x] **1.6** Tests (`tests/unit/test_benchmark_negatives.py`, 7) + full suite green; locked gate + smoke test unaffected.

### Baseline established (first measurement)

| Signal | Value | Note |
|---|---|---|
| **false_answer_rate** | **91.7%** (11/12) | Retriever almost never abstains on out-of-corpus queries — the headline weakness. Motivates Phase 5. |
| extended hard P@1 / R@3 | 60% / 60% (3/5) | Real headroom vs the locked set pinned at 100%. |

## Phase 2 — Measure generalization

**Status**: complete (2026-06-07) — advisory, no gate changes.

- [x] **2.1** Unseen second corpus `tests/golden/corpus_b/` (8 raw markdown docs, a job-scheduler doc set; NO curated summaries/aliases — the realistic "point at an arbitrary docs folder" case).
- [x] **2.2** Generalization golden set `tests/golden/queries_corpus_b.yaml` (11 queries: easy/medium + **hard paraphrases** with deliberate vocabulary mismatch — the stress case for token-overlap retrieval).
- [x] **2.3** `_run_benchmark(..., corpus=)` now accepts a corpus; `--corpus` / `--corpus-queries` run a generalization pass via `DirectoryCorpus`, reported under `generalization` (advisory; never gates).
- [x] **2.4** Folded in the Phase 1 coverage polish that missed the #161 merge (removed untested faithfulness-branch dup, simplified the aggregate guard, added the branch tests).
- [x] **2.5** Tests: real-retriever generalization test + main JSON wiring.

### Findings — the generalization gap is real

| Metric | attune-help (tuned) | corpus_b (unseen) |
|---|---|---|
| precision@1 | 97.5% | **72.7%** |
| recall@3 | 100% | **81.8%** |
| hard tier | — | **50%** (2/4) |

Keyword+alias retrieval drops ~25pts on an unseen corpus, and **paraphrase
queries (hard tier) fail half the time** — one returned nothing at all.
**This is the data that gates Phase 3 in: hybrid/embedding retrieval is
justified** for arbitrary user corpora (it would close the paraphrase gap
that token overlap can't).

### Hard-tier follow-up diagnostic — measured-out (2026-06-07)

After Phases 3/5 shipped, the residual question was: can we lift the
corpus_b **hard tier** (paraphrase, 50% P@1) further? Diagnostic traced
all 4 hard queries, then ran an **LLM-query-expansion ceiling test**
(Haiku rewrites each lay paraphrase into doc vocabulary, then keyword
retrieval — ~4 LLM calls):

| corpus_b hard tier (n=4) | P@1 | R@3 |
|---|---|---|
| keyword / hybrid (baseline) | 0.50 | 0.50–0.75 |
| **+ LLM query expansion (ceiling)** | **0.75** | **0.75** |

**Verdict: don't build hard-tier lift into the offline/deterministic core.**

- **Offline levers structurally can't close it.** The failing queries have
  *zero lexical overlap* with their target docs ("dying"↔"dead-letter",
  "move back"↔"requeue"). Thesaurus expansion can't bridge domain jargon;
  doc-side alias inference can't add words the doc never uses.
- **The strongest lever only recovers half, via the gated-out dependency.**
  LLM expansion took hard-tier 50%→75% — but that is the same
  API/network/`[claude]` dependency **data-gated out for rerank**
  ([#165](https://github.com/Smart-AI-Memory/attune-rag/pull/165)). Same
  tradeoff, same verdict.
- **The residual miss is irreducible.** cb-011 stays wrong even with ideal
  vocabulary (its expansion contained the target's exact terms), because a
  longer sibling doc legitimately out-scores the short target on shared
  terms — concept-overlap ambiguity, not a retrieval defect.
- **The metric is n=4** — "50%→75%" is one query (cb-008). Too small and
  noise-dominated to justify dependency-adding infra.

Future home if ever justified: an **opt-in LLM query-expansion companion to
the opt-in reranker (v1.1.0+, same dependency class)** — not the v1.0.0
default. **Reopen trigger:** a real user corpus + a ≥30-query hard set
showing a sustained, *offline-reachable* gap.

## Phase 3 — Hybrid retrieval

**Status**: complete (2026-06-07).

- [x] **3.1** `[embeddings]` extra = `model2vec` (static/distilled embeddings: no torch, offline, ms-encode). KeywordRetriever stays the default.
- [x] **3.2** `EmbeddingRetriever` (cosine over static embeddings; per-corpus matrix cache; injectable encoder for tests; numpy/model2vec imported lazily so the module is import-safe without the extra).
- [x] **3.3** `HybridRetriever` — keyword + embedding fused via **weighted RRF**; graceful fallback to keyword-only when the extra is absent.
- [x] **3.4** Benchmark `--retriever {keyword,hybrid}` to measure either.
- [x] **3.5** CI installs `[dev,embeddings]` so the new code is exercised (fake-encoder tests; no model download). 100% coverage on both new modules.

### Findings — the lift is real, but corpus-dependent (no free lunch)

| | attune-help (tuned) | corpus_b (unseen) |
|---|---|---|
| keyword (default) | P@1 100% / R@3 100% | P@1 73% / R@3 82% |
| hybrid, naive RRF (1:1) | P@1 **80%** / R@3 100% | P@1 82% / R@3 91% |
| hybrid, keyword-favoring (≥5:1) | P@1 **100%** / R@3 100% | P@1 73% / R@3 **91%** |

- **recall@3 +9pts on the unseen corpus holds at every weight** — and at a keyword-favoring weight it costs **zero** regression on the tuned corpus. Since RAG feeds top-*k*, recall@3 is the metric that matters.
- Equal-weight RRF trades away tuned-corpus top-1 precision (100→80). So: **keyword stays the pipeline default**; hybrid is **opt-in** with a keyword-favoring default weight (2:1) and tunable `keyword_weight`/`embedding_weight`.

**Decision:** ship hybrid as an opt-in tool for unstructured/arbitrary corpora; don't change the default.

## Phase 4 — Rerank stage (data-gated)

**Status**: closed — **data-gated out** (2026-06-07). The reranker code (`src/attune_rag/reranker.py`, `LLMReranker`) and the pipeline over-fetch path already exist; the gate measurement says **don't promote it** to a shipped/tuned stage. No PR.

- [x] **4.1** Gate measurement: P@1 / R@3 for keyword vs hybrid vs hybrid+rerank on BOTH the tuned corpus (attune-help, `queries.yaml`, n=40) and the unseen corpus (corpus_b, `queries_corpus_b.yaml`, n=11). `LLMReranker` = Claude Haiku 4.5, `candidate_multiplier=3`. ~51 LLM calls — the only API-spending step in the whole program.

### Findings — rerank does not earn its dependency

| corpus | config | P@1 | R@3 | hard P@1 | hard R@3 |
|---|---|---|---|---|---|
| attune-help (tuned) | keyword | **1.000** | 1.000 | **1.000** | 1.000 |
| attune-help | hybrid | 0.850 | 1.000 | 0.667 | 1.000 |
| attune-help | hybrid+rerank | 0.900 | 1.000 | 0.667 | 1.000 |
| corpus_b (unseen) | keyword | 0.727 | 0.818 | 0.500 | 0.500 |
| corpus_b | hybrid | 0.727 | **0.909** | 0.500 | **0.750** |
| corpus_b | **hybrid+rerank** | 0.727 | 0.909 | 0.500 | 0.750 |

- **On the unseen corpus (the generalization target that matters most), rerank gives ZERO lift over hybrid — every metric identical** (P@1, R@3, hard P@1, hard R@3). Hybrid's RRF already maxed what's reachable here.
- On the tuned corpus, rerank only *partially repairs* hybrid's own top-1 regression (0.85→0.90) and still lands **below plain keyword (1.000)**; R@3 was already 1.000 with no reranker. Hard-tier P@1 is unmoved (0.667).
- Rerank's lever is top-1 ordering, but RAG feeds top-*k* — R@3 is the metric that matters, and rerank moved it nowhere on either corpus.
- Cost of promoting it: the `[claude]` extra + an `ANTHROPIC_API_KEY` + network + per-query Haiku latency (~1.2s/query: 49s for 40 queries vs <1s keyword/hybrid) + token spend — for **no measured recall gain**.

**Decision:** close Phase 4 — do **not** build/tune a rerank stage. `LLMReranker` stays in-tree as a defined, opt-in escape hatch (`RagPipeline(reranker=LLMReranker())`) for future arbitrary corpora, but it is **not** promoted, not a default, and not part of the shipped retrieval story. This was the last open phase → **the rag-strengthening program is DONE** (Phases 1, 2, 3, 5 merged; Phase 4 data-gated out).

## Phase 5 — Faithfulness / abstention hardening

**Status**: complete (2026-06-07).

- [x] **5.1** Configurable abstention: `KeywordRetriever(min_score=T)` — when every candidate is below T the retriever returns nothing (abstains) instead of surfacing a weak match. Instance-level; default (2.0) unchanged.
- [x] **5.2** Calibration tool: `attune-rag-benchmark --calibrate-abstention` sweeps T over the legit (`--queries`) + out-of-corpus (`--negatives`) sets and recommends `min_score=T` for THIS corpus (max negatives-abstained s.t. legit-kept ≥ 95%). The threshold is an absolute keyword score, so it must be calibrated per corpus.
- [x] **5.3** Re-measured against the Phase 1 negative set (the hallucination/out-of-corpus set).
- [x] **5.4** Tests (`tests/unit/test_abstention.py`, 8) + README/CHANGELOG.

### Findings — the 91.7% false-answer rate is fixable

Top-1 scores separate cleanly on attune-help (legit median **14.8** vs out-of-corpus median **3.4**):

| min_score | legit kept | negatives abstained → false-answer rate |
|---|---|---|
| 2.0 (old default) | 100% | 8% → **92%** |
| **5 (calibrated)** | **98%** | **92% → 8%** |

`min_score=5` cuts the **false-answer rate 91.7% → 8%** for a 2pt legit-recall cost. Shipped as opt-in (`KeywordRetriever(min_score=…)`) with the calibration tool to pick T per corpus — **the default stays 2.0** (no behavior change).

## Notes

- `queries.yaml` is the **SHA-locked gate set** (`thresholds.json.queries_sha256`); never edit it without a threshold re-lock (see release-quality-baseline). New eval material goes in advisory side-files until promoted via a deliberate re-baseline.
