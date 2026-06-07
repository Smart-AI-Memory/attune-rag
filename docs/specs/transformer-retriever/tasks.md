# Spec: transformer-retriever — tasks

> **Status:** scoping (2026-06-07). M1+ not executable until the entry
> gate ([`requirements.md`](requirements.md#entry-gates)) opens. Likely
> v1.1.0+ (heavyweight opt-in; no default change). Scoping decisions
> filled at the `/spec` pass.

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

### M4 — Optional real-model CI — deferred (non-gating; not required for the tier)
- [ ] Non-gating optional job exercising the real model (kept off the
      core suite). Follow-up.

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
