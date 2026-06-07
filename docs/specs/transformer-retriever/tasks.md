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

### M0 — Reopen + sequence
- [ ] Record the narrow reopen in
      [`embedding-retriever`](../embedding-retriever/) (status note → here:
      torch returns as an opt-in tier, not a default).
- [ ] Confirm v1.1.0+ placement (or earlier opt-in) with `v1.0.0-release`.

### M1 — Second-corpus + asymmetric validation (the gate)
- [ ] Promote `torch_ceiling.py` / `torch_tsweep.py` to `scripts/`.
- [ ] Add a **second arbitrary corpus** + ≥30 hard queries (advisory
      side-files; never touch SHA-locked `queries.yaml`).
- [ ] Reproduce transformer-vs-torch-free margin on it (R1/R3).
- [ ] Measure symmetric vs asymmetric (query-prefix) encoding (design §5).
- [ ] **Gate:** margin holds on a second corpus? **No → close spec**
      (torch was a one-corpus artifact). **Yes → M2.**

### M2 — Implement the `[transformers]` tier
- [ ] `[transformers]` extra (torch + sentence-transformers, pinned per
      Q4); lazy import.
- [ ] Transformer backend (class or `backend=`, per Q1) reusing
      `EmbeddingRetriever`'s injectable-encoder path; default model per Q2;
      asymmetric encoding per Q3.
- [ ] Prove R2 (base install unchanged, no torch in default path) + R5
      (determinism) in CI with a **fake encoder** — no torch download in CI.

### M3 — Footprint, latency, offline, docs
- [ ] Disclose install-size + latency deltas (R4); document one-time
      download + offline/pre-cache path (R6).
- [ ] Operating-point guide (design §6): when to choose keyword vs static
      hybrid vs transformer.

### M4 — Optional real-model CI
- [ ] Non-gating optional job exercising the real model (kept off the
      core suite).

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
