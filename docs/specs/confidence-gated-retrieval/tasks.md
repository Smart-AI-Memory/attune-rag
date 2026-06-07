# Spec: confidence-gated-retrieval — tasks

> **Status:** scoping (2026-06-07). M1+ are **not** executable until the
> entry gate ([`requirements.md`](requirements.md#entry-gates)) opens.
> Scoping decisions are filled at the `/spec` pass.

## Scoping decisions (locked at `/spec` — TBD)

From [`design.md` §8](design.md#8-open-questions-for-scoping) /
[`requirements.md` "Open questions"](requirements.md#open-questions-for-scoping):

1. Hard switch vs below-T RRF blend — _TBD_ (decided by M2)
2. Gate signal: top-1 score vs top1−top2 gap — _TBD_
3. `GatedRetriever` class vs `gate=`/`mode=` on `HybridRetriever` — _TBD_
4. Shared calibration tool with `safe-abstention-defaults` vs linked — _TBD_
5. Default model: `potion-retrieval-32M` vs larger static — _TBD_

## Milestones

### M0 — Entry + reopen
- [ ] Record the reopen in [`embedding-retriever`](../embedding-retriever/)
      (status note → here).
- [ ] Confirm joint design with `safe-abstention-defaults` (shared signal).

### M1 — Build + validate the ≥30-query hard set (the gate)
- [ ] Author a **≥30-query paraphrase/hard set** (expand corpus_b and/or
      add a second arbitrary corpus) as an advisory side-file (never
      touch SHA-locked `queries.yaml`).
- [ ] Promote `gated_fusion.py` / `static_levers.py` to `scripts/`.
- [ ] Re-measure keyword vs embedding-only vs gated fusion on the ≥30-set.
- [ ] **Decision gate (R1):** does the gated lift hold at n≥30 with zero
      attune-help regression? **No → close this spec** (torch becomes the
      next question). **Yes → proceed to M2.**

### M2 — Mechanism bake-off
- [ ] Measure hard-switch vs below-T RRF blend (Q1), and gate-on-score vs
      gate-on-gap (Q2), on the ≥30-set incl. medium tier.
- [ ] Lock the rule + gate signal.

### M3 — Implement (the build PR)
- [ ] `GatedRetriever` (or `HybridRetriever(gate=…)`, per Q3), opt-in
      under `[embeddings]`, default model per Q5.
- [ ] Prove R2 (help 100/100), R3 (hard lift), R5 (base install
      unchanged) in CI. Disclose footprint delta (risk §2).
- [ ] `### Added`/`### Changed` per freeze decision (Q3 / R-NFR).

### M4 — Shared calibration + threshold
- [ ] Per-corpus T via the **shared** abstention/gate calibration (R4,
      Q4). One threshold, one tool, reproducible (R6).

### M5 — Docs
- [ ] README/onboarding: when to enable gated retrieval; footprint; the
      one "do I trust this retrieval?" decision tree (rescue vs abstain).

## Done when

- The unseen-corpus hard-tier lift is real at **n≥30** (not a 4-query
  artifact), torch-free and deterministic.
- attune-help stays 100% / 100% (zero regression), base install
  unchanged.
- One shared keyword-confidence threshold governs both rescue and
  abstention.
- The `embedding-retriever` reopen is recorded and the calibration is
  reproducible.

## Provenance

Opened 2026-06-07 from the
[`rag-strengthening` hard-tier amendment](../rag-strengthening/tasks.md)
after confidence-gated fusion (keyword-primary, `potion-retrieval-32M`
rescue, T=3) hit the LLM-expansion ceiling (hard P@1 0.75) with zero
attune-help regression — torch-free, no API. Reopens
[`embedding-retriever`](../embedding-retriever/); designed jointly with
[`safe-abstention-defaults`](../safe-abstention-defaults/).
