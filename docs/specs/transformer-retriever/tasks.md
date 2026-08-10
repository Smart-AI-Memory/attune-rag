# Spec: transformer-retriever — tasks

> **Status:** **complete (2026-08-10).** M0–M3 shipped 2026-06-07
> (freeze-override, Patrick per-PR); tree-verified 2026-08-10
> (`transformer.py`, `test_transformer.py`, `[transformers]` extra,
> README tier docs all present). M4 (non-gating real-model CI) shipped
> 2026-08-10 as a manual-dispatch workflow — see [`tasks.md`](tasks.md).

## Scoping decisions (locked at `/spec` — TBD)

From [`design.md` §8](design.md#8-open-questions-for-scoping):

1. `TransformerRetriever` class vs `EmbeddingRetriever(backend=…)` — _TBD_
2. Default model: `bge-small` (P@1) vs `MiniLM` (R@3/footprint) — _TBD_
3. Asymmetric query-prefix encoding — _TBD_ (adopt if M1 shows lift)
4. `[transformers]` pin: torch floor, CPU-only wheel — _TBD_
5. v1.1.0+ vs earlier opt-in add — _TBD_ (freeze)

## Milestones

> **Status: IMPLEMENTED 2026-06-07** (freeze-override authorized by Patrick
> per-PR). Scoping decisions locked by the M1 measurement: Q1 → a
> `TransformerRetriever` subclass of `EmbeddingRetriever`; Q2 → default
> `BAAI/bge-small-en-v1.5`; Q3 → **adopt** asymmetric query-prefix (it
> measured +5pts); Q4 → `sentence-transformers>=3.0,<6.0`; Q5 → ships as an
> opt-in extra now (no default change), freeze-override applied.

### M0 — Reopen + sequence
- [x] Narrow reopen of [`embedding-retriever`](../embedding-retriever/)
      recorded (torch returns as an opt-in tier, not a default).
- [x] Ships as an opt-in extra (no default change); v1.1.0+ pinning is moot.

### M1 — Second-corpus + asymmetric validation (the gate) — **PASS**
- [x] Promoted to `scripts/validate_transformer_retriever.py`.
- [x] Added a **second arbitrary corpus** (`tests/golden/corpus_c/`, HTTP
      API client — different domain/jargon) + `queries_corpus_c_hard.yaml`
      (24 queries, 20 hard). Never touches SHA-locked `queries.yaml`.
- [x] Reproduced the margin: corpus_c hard P@1 — keyword **0.25**, static
      **0.55**, transformer **0.85–0.90**, R@3 → **1.00**. Generalizes
      (corpus_b was 0.50→0.69; corpus_c 0.55→0.90).
- [x] Asymmetric (query-prefix) encoding measured **+5pts** (0.85→0.90) →
      adopted as the default for BGE.
- [x] **Gate PASS** → proceeded to M2.

### M2 — Implement the `[transformers]` tier — **DONE**
- [x] `[transformers]` extra = `sentence-transformers>=3.0,<6.0`
      (`pyproject.toml`, added to `all`); `uv.lock` regenerated. Lazy
      import in `TransformerRetriever._get_encoder`.
- [x] `TransformerRetriever(EmbeddingRetriever)` (`src/attune_rag/transformer.py`)
      reusing the matrix cache + cosine path; default `bge-small`; new
      `query_prefix` asymmetric hook on `EmbeddingRetriever`. Exported in
      `__init__`/`__all__`.
- [x] R2/R5 proven with a **fake encoder** (`tests/unit/test_transformer.py`,
      6 tests) — no torch download in CI. Base install unchanged (lazy
      import). Real end-to-end smoke reproduced 0.90/1.00.

### M3 — Footprint, latency, offline, docs — **DONE**
- [x] CHANGELOG `### Added` + README "Transformer retrieval" section with
      footprint (~GB torch), latency (~10–300 ms/query), one-time
      download/offline note, and the operating-point guide (keyword vs
      static hybrid vs transformer).

### M4 — Optional real-model CI — **DONE 2026-08-10**
- [x] Non-gating optional job exercising the real model (kept off the
      core suite): `.github/workflows/real-model-validation.yml`,
      **workflow_dispatch only** (never on push/PR, never required —
      non-gating by construction; no API key, retrieval-only). Runs
      `scripts/validate_transformer_retriever.py` (downloads bge-small)
      AND `scripts/measure_gated_mechanism.py` (ret-32M) so one dispatch
      re-validates BOTH heavyweight-adjacent tiers against drift
      (sentence-transformers/torch/model2vec version movement — risk §4).
      Receipt: both scripts run locally against the current tree
      (torch 2.12.0) before the workflow landed.

### Post-close correction note (2026-08-10)

The "torch-free ceiling ~0.50" framing that anchored this spec moved:
[`confidence-gated-retrieval` M2](../confidence-gated-retrieval/tasks.md)
measured the below-gate 1:1 RRF blend at **corpus_c hard P@1 0.70**
(corpus_b stays 0.50). The transformer's margin is intact and still
uniquely torch (corpus_b 0.50→0.69, corpus_c 0.70→0.90 — both ~+20pts
over the best torch-free config), but "≈0.50 ceiling" claims are now
corpus_b-only. README's transformer section refreshed accordingly.

## Done when

- An opt-in `[transformers]` retriever delivers the measured paraphrase
  quality (hard P@1 ≫ torch-free) on ≥2 arbitrary corpora.
- Base install, keyword default, and `[embeddings]` behavior are
  byte-for-byte unchanged; no torch in any default path.
- Footprint/latency/offline costs are documented; the core test suite
  never depends on torch.

## Provenance

Opened 2026-06-07 from the
[`confidence-gated-retrieval` M1b torch comparison](../confidence-gated-retrieval/tasks.md):
real transformers (bge-small 0.69 hard P@1, MiniLM 0.92 R@3) exceed the
~0.50 torch-free ceiling that keyword/static/gated all hit — the one goal
that uniquely needs sentence-transformers. Narrowly reopens
[`embedding-retriever`](../embedding-retriever/) as an opt-in tier.
