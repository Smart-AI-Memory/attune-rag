# Spec: safe-abstention-defaults (attune-rag)

> **Status:** **executing — scoping locked 2026-07-17** (M0–M2 done,
> decisions recorded in [`tasks.md`](tasks.md); measurements committed
> in [`measurements.md`](measurements.md)). Next: M3, the
> bundled-default behavior PR. The default-behavior change is **not**
> shipped by the measurement PR.
>
> Provenance: the [`rag-strengthening`](../rag-strengthening/) program
> (Phases 1/2/3/5 merged; Phase 4 closed data-gated-out,
> [#165](https://github.com/Smart-AI-Memory/attune-rag/pull/165))
> built the abstention *mechanism* but left it **opt-in**. This spec
> closes the gap between "we built the fix" and "the default is safe."

## Purpose

attune-rag v1.0.0 is framed as a **deterministic retrieval framework
for your own markdown corpus, with attune-help as the bundled
exemplar**. For that framing, *the default behavior is the product* —
most users get whatever ships out of the box.

The out-of-the-box behavior today has a measured weakness:

- [`rag-strengthening` Phase 1](../rag-strengthening/tasks.md) exposed a
  **91.7% false-answer rate** on out-of-corpus queries against the
  bundled corpus — the retriever almost never abstains.
- [Phase 5](../rag-strengthening/tasks.md) built the fix —
  `KeywordRetriever(min_score=T)` abstains when every candidate is
  below `T`, and `attune-rag-benchmark --calibrate-abstention`
  recommends `T` per corpus. `min_score=5` cut the bundled-corpus
  false-answer rate **92% → 8%** for a 2-point legit-recall cost.
- **But the shipped default is still `min_score=2.0`** — so a v1.0.0
  user still gets the 91.7% behavior unless they opt in.

This spec decides **what the safe default should be** and how to ship
it without pretending a single universal threshold exists (the
measurement below proves it doesn't).

## The load-bearing measurement (2026-06-07)

Top-1 keyword-score distributions, legit vs out-of-corpus, on both the
bundled corpus and the unseen `corpus_b` (probe:
[`design.md` §2](design.md#2-the-measurement-that-anchors-this-spec)):

| statistic (median) | attune-help legit | attune-help neg | corpus_b legit | corpus_b neg |
|---|---:|---:|---:|---:|
| **raw top-1** | 14.75 | 3.38 | **6.00** | **0.00** |
| top-1 / query-token | 4.30 | 0.58 | 1.67 | 0.00 |
| top-1 − top-2 gap | 5.25 | 0.68 | 4.00 | 0.00 |

Three findings drive every decision in this spec:

1. **The false-answer problem is corpus-shape-specific, not universal.**
   attune-help is alias/vocabulary-rich, so out-of-corpus queries leak
   spurious matches at **median 3.38 — above the 2.0 default**. On a
   lean corpus (`corpus_b`) out-of-corpus queries score **~0** and the
   *current* default already abstains correctly. The headline weakness
   lives on the bundled exemplar specifically.

2. **A single global `min_score` is provably unsafe.** `min_score=5`
   fixes attune-help — but `corpus_b`'s *legit* median top-1 is only
   **6.0**, so a global 5 starts cutting real corpus_b answers. "Just
   bump the default" is dead on arrival, with data.

3. **The cleanest corpus-agnostic signals are per-token-normalized
   score (7.4× separation) and the top1−top2 gap (7.8×)** — both far
   better than raw top-1 (4.4×) on attune-help — but they still vary
   ~2.6× across corpora, so they are a strong *heuristic floor*, not a
   universal constant.

## What ships (eventually — not in this scaffold)

A v1.0.0 default-behavior change, shaped by the findings above:

- **Bundled attune-help** ships with its *known, calibrated* abstention
  default baked in (the measured 92%→8% fix). We have the data for the
  exemplar; the default should reflect it.
- **Bring-your-own corpora** get **auto-calibration** when the user
  supplies eval data, or a **calibrate-first onboarding step** when
  they don't — never a magic universal number.
- **Optional** a per-token / top1−top2-gap relative heuristic as a
  calibration-free *floor* for corpora with no eval data.

The deliverable of *this spec* is the scaffold (the four files below)
plus the locked scoping decisions. The behavior change is a separate
PR at the v1.0.0 cut.

## What's *not* in scope

- **Editing `queries.yaml`.** It is the SHA-locked gate set
  (`thresholds.json.queries_sha256`); abstention calibration reads it
  but never mutates it. New eval material goes in advisory side-files.
- **A new retriever or scoring-model change.** This spec tunes the
  *abstention threshold*, not how scores are computed. Scoring stays as
  shipped.
- **Reopening rerank or embeddings defaults.** Rerank is data-gated-out
  ([#165](https://github.com/Smart-AI-Memory/attune-rag/pull/165));
  hybrid stays opt-in. This spec is orthogonal to both.
- **A faithfulness floor in POLICY.** Untouched; see
  [`feedback_policy_llm_metric_commitments`](../../../).

## Relationship to other specs

- **[`user-corpus-onboarding`](../user-corpus-onboarding/)** — the
  natural home for the *BYO calibrate-first* path. This spec owns the
  *bundled-default* decision and the calibration *mechanism*; the
  onboarding guide consumes it. Cross-link must land before either
  closes scoping, to avoid two surfaces disagreeing (the same hazard
  [`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md)
  flagged for the reranker default).
- **[`v1.0.0-release`](../v1.0.0-release/)** — the default-change PR is
  a v1.0.0 cut item; sequence it there.
- **POLICY §7** — the bundled default change interacts with the
  retrieval-quality floor commitment; see
  [`risks.md`](risks.md).

## Layout

- [`requirements.md`](requirements.md) — invariants the default-change
  must satisfy (no legit-recall regression on the bundled gate set,
  freeze/POLICY compatibility, per-corpus honesty, reproducibility).
- [`design.md`](design.md) — the measurement that anchors the spec, the
  three candidate mechanisms (baked-in bundled default / auto-calibrate
  / relative-heuristic floor) with trade-offs, and the open scoping
  questions.
- [`risks.md`](risks.md) — over-abstention on lean corpora, the
  default-flip-during-freeze risk, POLICY §7 interaction, negatives-set
  representativeness.
- [`tasks.md`](tasks.md) — milestones from entry-gate through the
  default-change PR; scoping questions answered at the `/spec` pass.
