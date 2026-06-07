# Spec: RAG Strengthening

## Phase 1: Requirements

**Status**: approved (2026-06-06)

### Problem statement

attune-rag is excellent on its home turf and blind everywhere else. The
eval gate sits at ~100% recall@3 on a single bundled corpus (attune-help),
and the embedding retriever was deliberately deferred because keyword +
alias-expansion already nails that corpus. Two consequences:

1. **The gate can't detect regressions** — metrics are pinned at ceiling,
   so degradation is invisible.
2. **Zero signal on arbitrary/user corpora** — the deferral's own caveat.
   "Stronger" is currently unmeasurable.

This spec strengthens attune-rag by **measuring before adding mechanisms**,
then adding retrieval/faithfulness improvements validated against the
hardened measurement.

### The program (5 phases, sequenced)

| # | Phase | Why | Gated on |
|---|-------|-----|----------|
| 1 | **Harden the eval** | A gate at 100% is a smoke alarm with no battery. Add negative/abstention cases, distractors, per-difficulty reporting, headroom. | — |
| 2 | **Measure generalization** | Add a second, unseen corpus track so we know if keyword+alias is overfit to attune-help vocabulary. | Phase 1 metrics |
| 3 | **Hybrid retrieval** (`[embeddings]` extra) | Revive the deferred embedding retriever + RRF fusion for vocabulary mismatch — *only if* Phase 2 shows keyword-only underperforms. | Phase 2 data |
| 4 | **Rerank stage** | Optional cross-encoder/LLM reranker over top-k to lift precision on harder corpora. | Phase 1/2 metrics |
| 5 | **Faithfulness/abstention hardening** | Explicit abstention when retrieval confidence is low; hallucination test set. Makes the library trustworthy downstream. | Phase 1 negative set |

Phases 3 and 4 are **data-gated**: build only if the measurement from
1–2 shows the need. This avoids speculative complexity.

### Phase 1 scope (this PR)

**In scope:**

- **Negative / out-of-corpus query set** (`tests/golden/queries_negative.yaml`):
  questions the bundled corpus genuinely cannot answer (out-of-domain, and
  near-domain-but-absent). Correct behavior = the retriever **abstains**
  (returns no hit above `MIN_SCORE`).
- **New measured metrics** in `benchmark.py`:
  - `false_answer_rate` — fraction of negative queries that returned ≥1 hit
    (lower is better; this is the current blind spot, measured for the first
    time).
  - `abstention_rate` — `1 − false_answer_rate`.
  - **Per-difficulty breakdown** of precision@1 / recall@k (easy/medium/hard),
    so a regression on hard queries isn't masked by easy ones at ceiling.
- **More hard queries** in `queries.yaml` to create recall headroom (the set
  is currently 10 easy / 27 medium / 3 hard).
- Tests for the new aggregation + negative pass.

**Explicitly NOT in scope (Phase 1):**

- **Gating** on the new metrics. Phase 1 is *measurement*: report the
  metrics, establish a baseline. Locking CI thresholds on them is a
  follow-up once we've observed stable values over N runs (mirrors the
  release-quality-baseline variance methodology). No new red-CI risk.
- Changing the retriever's scoring or `MIN_SCORE` (that's Phase 5 territory
  if the false-answer baseline is bad).
- Embeddings / rerank (Phases 3–4).

### Acceptance criteria (Phase 1)

- `queries_negative.yaml` exists with ≥10 out-of-corpus questions.
- `benchmark.py` reports `false_answer_rate`, `abstention_rate`, and
  per-difficulty precision/recall; printed in the summary and present in the
  JSON dump.
- ≥5 new hard queries added to `queries.yaml`.
- New unit tests cover the negative pass and per-difficulty aggregation.
- Full suite green; the existing locked gate (precision@1 / recall@3 /
  faithfulness) is unaffected.

### Affected layers

- [x] attune-rag (eval harness + golden sets)
- [ ] attune-ai / attune-author / attune-gui — no changes
