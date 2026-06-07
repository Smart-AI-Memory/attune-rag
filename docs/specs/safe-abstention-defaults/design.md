# Spec: safe-abstention-defaults — design

> **Status:** scoping (2026-06-07). Options below are candidates;
> the chosen mechanism is locked at the `/spec` pass and recorded at
> the top of [`tasks.md`](tasks.md).

## 1. The problem in one paragraph

`KeywordRetriever` keeps any candidate scoring `>= MIN_SCORE` (default
**2.0**) and abstains only when *every* candidate falls below it. The
threshold is an **absolute** keyword score, so its correct value
depends on the corpus's vocabulary/alias density. The default of 2.0
was never calibrated; on the alias-rich bundled corpus it lets
out-of-corpus queries through at a 91.7% rate. We need a default story
that is safe across corpus shapes without hardcoding a constant that
can only be right for one corpus.

## 2. The measurement that anchors this spec

Probe (`/tmp/abstain_probe.py` at authoring time; promote to
`scripts/measure_abstention_distributions.py` at M1): with
`KeywordRetriever(min_score=0)` so nothing is filtered, take the top-1
hit per query and record raw score, score ÷ query-token-count, and the
top1−top2 gap. Run over legit queries and the out-of-corpus negative
set, on both corpora.

| statistic (median) | help legit | help neg | corpus_b legit | corpus_b neg |
|---|---:|---:|---:|---:|
| raw top-1 | 14.75 | 3.38 | 6.00 | 0.00 |
| top-1 / query-token | 4.30 | 0.58 | 1.67 | 0.00 |
| top-1 − top-2 gap | 5.25 | 0.68 | 4.00 | 0.00 |

Separation ratios (legit median ÷ neg median) on attune-help:
raw top-1 **4.4×**, per-token **7.4×**, gap **7.8×**.

Reading of the data:

- **Negatives behave very differently per corpus.** On attune-help,
  out-of-corpus queries score a median **3.38** — *above* the 2.0
  default, which is exactly why the false-answer rate is 91.7%. On
  corpus_b they score **~0** (no token overlap at all), so the existing
  default already abstains. The bundled exemplar is the hard case.
- **Legit magnitude is also corpus-dependent.** attune-help legit
  top-1 median **14.75** vs corpus_b **6.00** — a 2.5× difference
  driven by alias/vocabulary density. This is what makes a global
  absolute threshold unsafe: 5 is comfortably below attune-help's
  legit floor but bites into corpus_b's.
- **Normalizing helps but does not universalize.** Per-token and gap
  both separate legit/neg much more cleanly than raw score *within* a
  corpus, but their legit medians still differ ~2.6× *across* corpora.
  So they make a better calibration-free *floor* than raw score —
  not a substitute for per-corpus calibration.

## 3. Candidate mechanisms

These are not mutually exclusive; the likely answer is **C1 for the
bundled corpus + one of C2/C3 for BYO**.

### C1 — Bake the calibrated default into the bundled corpus

Ship attune-help with its measured threshold (≈5) as *its* default,
rather than relying on the class-level `MIN_SCORE`. Mechanism options:

- A per-corpus `min_score` carried on the bundled corpus metadata /
  pipeline factory, so `RagPipeline()` (bundled) gets 5 while a raw
  `KeywordRetriever()` keeps the conservative class default.
- Or flip the class default and special-case nothing — **rejected** by
  §2: 5 is unsafe for lean corpora like corpus_b.

**Pro:** directly fixes the 91.7% headline for the exemplar everyone
sees first; we already have the calibration data. **Con:** behavior
change (POLICY §7 + freeze interaction — see risks); must prove no
legit-recall regression on the SHA-locked gate set.

### C2 — Auto-calibrate at corpus load when eval data exists

When the user supplies a legit query set (+ optional negatives),
derive `min_score` from the score distribution the same way
`--calibrate-abstention` does, and apply it automatically. No eval
data → fall back to a conservative default (C3 floor or current 2.0).

**Pro:** honest — uses the corpus's own data; reuses the Phase 5
calibration logic. **Con:** most users won't have a negatives set;
needs a clear "calibrate-first" onboarding path (→ `user-corpus-onboarding`).

### C3 — Relative / normalized heuristic floor (calibration-free)

Replace the raw-score gate with a normalized criterion that travels
better across corpora — candidates from §2:

- **per-token:** abstain if `top1 / n_query_tokens < τ`
- **gap:** abstain if `top1 - top2 < δ` (top-1 must stand out)
- **combination:** keep if either a modest absolute floor *or* a
  relative criterion is met.

**Pro:** needs no per-corpus data; corpus_b negatives at ~0 mean any
sane floor abstains correctly there; per-token/gap separate 7.4×/7.8×
on attune-help. **Con:** the ~2.6× cross-corpus variance means a fixed
τ/δ is still a heuristic, not a guarantee; risk of over-abstaining on
very short legit queries (single-token lookups). Must be measured on
both corpora's *legit* sets before adoption, not just negatives.

## 4. Recommended shape (for ratification at `/spec`)

1. **C1 for the bundled corpus** — bake in the calibrated default so
   the exemplar is safe out of the box. Gate it on *zero* regression
   against `tests/golden/queries.yaml` (P@1/R@3 must hold).
2. **C2 for BYO with eval data**, surfaced through
   `user-corpus-onboarding` as the recommended path.
3. **C3 as the no-eval-data fallback floor**, *only if* it clears a
   both-corpora legit-recall bar; otherwise BYO-without-data keeps the
   conservative current default and a loud doc pointer to
   `--calibrate-abstention`.

## 5. What this spec does NOT decide

- The exact numeric default for the bundled corpus — that is re-derived
  by the M1 calibration run against the then-current corpus, not
  hardcoded from this scaffold's snapshot.
- Whether C3's heuristic ships at v1.0.0 or defers to v1.1.0 — gated on
  its both-corpora legit-recall measurement (M2).
- Any change to the scoring model itself.

## 6. Open questions for scoping

1. Does the bundled default live on corpus metadata, the pipeline
   factory, or a new `KeywordRetriever` construction path? (API surface
   → freeze relevance.)
2. Is the bundled-default change a `### Changed` (like the #120 perf
   gate precedent) or does it need a `freeze-override`?
3. Does C3's heuristic clear a legit-recall bar on *both* corpora, or
   does it over-abstain on short queries?
4. Where does the auto-calibration entry point live —
   `RagPipeline.calibrated(corpus, queries, negatives)` classmethod, or
   inside the onboarding harness only?
5. What does POLICY §7 say after this lands — does the safe default
   become a stated behavioral commitment, or stay advisory?
