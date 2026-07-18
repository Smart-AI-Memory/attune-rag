# Spec: safe-abstention-defaults — requirements

> **Status:** executing — scoping locked 2026-07-17 (see
> [`tasks.md`](tasks.md)). The requirements below bind the
> **default-change PR** (M3).

## Entry gates

The scaffold lands today regardless of gate state. The
**default-change implementation** activates when:

1. **v1.0.0 cut is open for `/spec` scoping** (per
   [`v1.0.0-release`](../v1.0.0-release/)). A default-behavior change
   belongs at a major cut, not mid-freeze, unless explicitly
   freeze-overridden.
2. **`user-corpus-onboarding` M1 has not yet closed scoping**, OR the
   cross-link in §R4 is landed first — so the harness default and this
   spec's default cannot disagree.

## Functional requirements

### R1 — No legit-recall regression on the bundled gate set

The chosen bundled default MUST hold **P@1 = 100% and R@3 = 100%** on
`tests/golden/queries.yaml` (the SHA-locked gate set). The Phase 5
measurement showed `min_score=5` keeps 98% of legit *negatives-adjacent*
recall; this requirement is stricter — the gate set itself must not
regress at all. Verified by `attune-rag-benchmark` in CI.

### R2 — Measured false-answer reduction on the bundled corpus

The bundled default MUST reduce the out-of-corpus false-answer rate
materially below the 91.7% baseline (target: the Phase 5 ≈8% figure,
re-derived against the then-current corpus). Measured against
`tests/golden/queries_negative.yaml` via the existing
`_run_negative_benchmark` path. Advisory metric, but the PR must report
the number.

### R3 — Per-corpus honesty (no unsafe universal constant)

No single absolute `min_score` may be hardcoded as the global default
for *all* corpora. The design MUST distinguish the bundled corpus
(calibrated) from BYO corpora (auto-calibrate / conservative floor).
This requirement exists because the measurement
([`design.md` §2](design.md#2-the-measurement-that-anchors-this-spec))
proved `min_score=5` over-abstains on lean corpora (corpus_b legit
median top-1 = 6.0).

### R4 — Single source of truth for the default posture

The bundled default and any harness default surfaced by
[`user-corpus-onboarding`](../user-corpus-onboarding/) MUST derive from
one place. A cross-link landing before either spec closes scoping is
mandatory (mirrors the reranker-default hazard in
[`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md)).

### R5 — Reproducibility

The calibration that produces the bundled default MUST be reproducible
from a committed script (`scripts/measure_abstention_distributions.py`)
+ committed inputs (`queries.yaml`, `queries_negative.yaml`, corpus).
No hand-tuned magic numbers without a script that regenerates them. The
distribution probe in `design.md` §2 is the prototype for this script.

### R6 — Calibration never mutates the SHA-locked gate set

Calibration reads `queries.yaml` / negatives but never edits them.
Any new eval material lands in advisory side-files until promoted via a
deliberate re-baseline (per the SHA-lock convention).

## Non-functional requirements

- **Freeze compatibility.** The scaffold is docs-only. The
  default-change PR is either a `### Changed` (preferred, per the #120
  perf-gate precedent — it changes behavior, adds no public surface) or
  ships with a `freeze-override` if it must add a construction path.
  Decided at scoping (`design.md` §6 Q1/Q2).
- **Determinism.** Abstention must remain deterministic — same query +
  corpus → same answer/abstain decision. No randomness, no LLM in the
  default path (abstention stays keyword-only and offline).
- **Backward-compatible escape hatch.** Existing
  `KeywordRetriever(min_score=…)` explicit construction keeps working
  unchanged; the default change must not break callers who already pass
  a threshold.

## Open questions for scoping

Carried from [`design.md` §6](design.md#6-open-questions-for-scoping);
answered at the top of [`tasks.md`](tasks.md) at the `/spec` pass:

1. API surface for the bundled default (metadata / factory / new path).
2. `### Changed` vs `freeze-override`.
3. Does C3's relative heuristic clear a both-corpora legit-recall bar?
4. Auto-calibration entry-point location.
5. POLICY §7 wording after the change.
6. Cross-tier abstention (from the 2026-06-10 audit finding below):
   extend the safe default to the hybrid/transformer tiers, or scope
   abstention as keyword-tier-only and document the loss on upgrade?

## Audit input — abstention is keyword-tier-only (2026-06-10)

Independent confirmation from the 2026-06-10 usability audit that this
spec targets real user pain, plus a gap the scaffold didn't yet name:
the abstention mechanism exists **only on `KeywordRetriever`**.

- `EmbeddingRetriever` / `TransformerRetriever` always return top-k —
  cosine similarity has no floor; they can never abstain.
- `HybridRetriever` has no threshold of its own, and its keyword leg's
  `min_score` does not gate the fused result — an embedding-leg hit
  surfaces even when every keyword candidate was dropped.

Consequence: a user who adopts abstention and later upgrades a tier —
the README's own recommended path for paraphrase-heavy corpora —
**silently loses out-of-corpus protection**. The audit's step-2
remediation made the CLI contract explicit in the meantime:
`attune-rag query --min-score` combined with a non-keyword
`--retriever` is rejected with an error that names this spec. Open
question 6 above decides whether that rejection is the permanent
contract or a stopgap.
