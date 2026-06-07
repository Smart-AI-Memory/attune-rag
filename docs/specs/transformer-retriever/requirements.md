# Spec: transformer-retriever — requirements

> **Status:** scoping (2026-06-07). Scaffold is docs-only/freeze-clean.
> Requirements below bind the **implementation PR**.

## Entry gates

Scaffold lands now. Implementation activates when:

1. **R1 second-corpus validation passes** — the n=26 lift is single-corpus
   (corpus_b). Reproduce the transformer advantage on a *second* arbitrary
   corpus before building.
2. **The `embedding-retriever` defer is formally reopened** (status note
   → here): this tier is the narrow, opt-in, torch-back-as-a-tier reopen.
3. **Freeze allows it** — opt-in addition, no default change; ships at a
   cut window or with a `freeze-override` (design §8 Q5).

## Functional requirements

### R1 — Validate the transformer advantage on a second corpus

The 0.69 / 0.92 result is corpus_b-only. Before building, reproduce the
transformer-beats-torch-free margin on a **second arbitrary corpus**
(different domain). **Gate:** if the advantage doesn't generalize, this
spec closes — torch isn't justified for one corpus.

### R2 — Opt-in only; base install and all defaults unchanged

Installing without `[transformers]` MUST behave exactly as today.
Keyword stays the base default; static hybrid stays the `[embeddings]`
behavior. No import of torch in any default path. The transformer leg is
reachable only by explicit construction under the extra.

### R3 — Measured quality bar

On the R1 validation set, the transformer retriever (embedding-primary)
MUST beat the torch-free ceiling on hard-tier P@1 and/or R@3 by a margin
that holds at n≥30 (prototype: P@1 0.50→0.69, R@3 →0.92). Reported in the
PR.

### R4 — Footprint + latency disclosed and bounded

The PR MUST state the installed size delta (torch + model) and per-query
latency vs static/keyword. The default model MUST be a *small* retrieval
model (bge-small / MiniLM class), not a large one, and MUST be
configurable.

### R5 — Determinism

Encoding MUST be deterministic for a fixed model + input (eval mode, no
dropout, fixed dtype). Retrieval order MUST be stable (existing tie-break
by path holds). Document any cross-torch-version float drift.

### R6 — Offline after first download

Like `[embeddings]`, the model downloads once then runs offline. The PR
MUST document the one-time download and support a pre-cached / offline
install path (no network at query time).

## Non-functional requirements

- **Lazy import.** torch / sentence-transformers imported only inside the
  transformer backend, so the module stays import-safe without the extra
  (mirrors the current `model2vec` lazy import).
- **No coupling to defaults.** The tier must not change `RagPipeline()`
  default behavior or the `[embeddings]` hybrid default.
- **CI.** Tests use an injected fake encoder (as the static tests do) —
  **no torch download in CI**. A separate, optional, non-gating job may
  exercise the real model.
- **Freeze.** Adding the extra + backend is public surface → cut-window
  or `freeze-override` (design §8 Q5).

## Open questions for scoping

From [`design.md` §8](design.md#8-open-questions-for-scoping): class vs
backend flag; default model; asymmetric encoding; extra pinning; v1.1.0+
vs earlier opt-in.
