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

## Phase 3 — Hybrid retrieval (data-gated on Phase 2) → **GO**

- [ ] Phase 2 confirmed keyword-only underperforms on unseen corpora (esp. paraphrases). Revive the deferred embedding retriever behind a `[embeddings]` extra + RRF fusion; measure the lift against `corpus_b` (target: recover the hard-tier 50%).

## Phase 4 — Rerank stage (data-gated)

- [ ] Optional cross-encoder / LLM reranker over top-k; measure precision lift against Phase 1/2 sets.

## Phase 5 — Faithfulness / abstention hardening

- [ ] Drive `false_answer_rate` down: add a confidence/abstention threshold so out-of-corpus queries return nothing instead of a low-score hit. Add a hallucination test set. Re-measure against the Phase 1 negative set.

## Notes

- `queries.yaml` is the **SHA-locked gate set** (`thresholds.json.queries_sha256`); never edit it without a threshold re-lock (see release-quality-baseline). New eval material goes in advisory side-files until promoted via a deliberate re-baseline.
