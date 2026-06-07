# Spec: confidence-gated-retrieval — requirements

> **Status:** scoping (2026-06-07). Scaffold is docs-only and
> freeze-compliant. Requirements below bind the **implementation PR**.

## Entry gates

The scaffold lands now. Implementation activates when:

1. **R1's ≥30-query hard-set validation has passed** (below). No build
   on the n=4 signal.
2. **Designed jointly with [`safe-abstention-defaults`](../safe-abstention-defaults/)** —
   the shared-threshold requirement (R4) is settled before either ships.
3. **The `embedding-retriever` defer is formally reopened** in its spec
   (status note pointing here).

## Functional requirements

### R1 — Validate on a ≥30-query hard set before building

The n=4 corpus_b hard tier is too thin to justify new code. Before
implementation, build a **≥30-query paraphrase/hard set** (corpus_b
and/or a second arbitrary corpus) and reproduce the gated-fusion lift
on it. **Gate:** gated fusion must beat keyword on hard-tier P@1 by a
margin that holds at n≥30 — not a one-query artifact. If it does not,
this spec closes (and a torch model becomes the next question).

### R2 — Zero regression on the bundled corpus

The gated retriever MUST hold **P@1 = 100% / R@3 = 100%** on
`tests/golden/queries.yaml` (SHA-locked). The T=3 prototype already
does; the requirement is that the shipped config provably does, in CI.

### R3 — Measured hard-tier lift

The gated retriever MUST lift unseen-corpus hard-tier P@1 materially
above keyword (prototype: 0.50 → 0.75), measured on the R1 ≥30-query
set, reported in the PR.

### R4 — Single shared confidence threshold with abstention

The gate threshold and the abstention `min_score` are the **same
keyword-confidence signal** (design §4). They MUST derive from one
calibration and be mutually consistent per corpus — never two
independent thresholds that can disagree about whether a retrieval is
trustworthy. Enforced by a shared calibration tool or a shared
threshold value.

### R5 — Opt-in; keyword stays the default

Keyword-only remains the base-install default. The gated retriever is
opt-in under the `[embeddings]` extra. A base install (no extra) MUST
behave exactly as today. No default flip without a separate decision.

### R6 — Reproducible calibration

The per-corpus threshold MUST be reproducible from a committed script
(`scripts/`) + committed inputs. No hand-tuned T without a regenerating
script. The `gated_fusion.py` / `static_levers.py` probes are the
prototypes.

## Non-functional requirements

- **No torch.** The model stays a torch-free static distillation
  (`potion-retrieval-32M` class). A torch dependency is out of scope
  (README) unless R1 fails for static.
- **Determinism.** Same query + corpus → same hits. No randomness, no
  network, no LLM in the retrieval path.
- **Footprint disclosed.** `potion-retrieval-32M` is heavier than the
  current `potion-base-8M`; the PR MUST state the download/RSS delta and
  keep the model configurable so users can choose the lighter one.
- **Freeze.** Adding a retriever/class is public surface → either ships
  at the v1.0.0 cut or takes a `freeze-override` with rationale.
  Decided at scoping (`design.md` §8 Q3).

## Open questions for scoping

Carried from [`design.md` §8](design.md#8-open-questions-for-scoping):
hard-switch vs blend; gate signal (score vs gap); class vs option;
shared-vs-linked calibration; default model.
