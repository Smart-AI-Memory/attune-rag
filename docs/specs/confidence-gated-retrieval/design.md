# Spec: confidence-gated-retrieval — design

> **Status:** scoping (2026-06-07). Mechanism candidates below; the
> chosen shape is locked at the `/spec` pass and recorded atop
> [`tasks.md`](tasks.md).

## 1. Problem

Keyword retrieval is authoritative on lexically-aligned queries (the
tuned attune-help corpus: 100% P@1) but fails on paraphrases with no
token overlap (corpus_b hard tier: 50%). Embeddings are the inverse —
strong on paraphrase, weak on the alias-tuned corpus. The shipped
`HybridRetriever` fuses them with weighted RRF, but RRF lets the
keyword leg's confident-but-wrong top rank dominate, so it inherits
keyword's hard-tier failures (measured: hard P@1 0.50 at every weight).

## 2. The measurement that anchors this spec

All torch-free, deterministic. Target = LLM-expansion ceiling (0.75).

| config | hard P@1 | corpus_b P@1 / R@3 | help P@1 / R@3 |
|---|---:|---:|---:|
| keyword | 0.50 | 0.73 / 0.82 | 1.00 / 1.00 |
| RRF 2:1 / 1:1 / 1:2 / 1:3 (8M) | 0.50 | 0.73–0.82 / 0.91 | 1.00→0.53 |
| embedding-only potion-retrieval-32M | 0.75 | 0.91 / 0.91 | 0.28 / 0.68 |
| gated T=3 (kw-primary, 32M rescue) | **0.75** | 0.82 / 0.91 | **1.00 / 1.00** |
| gated T=6 | 0.75 | 0.91 / 0.91 | 0.95 / 0.97 |
| gated T=8 | 0.75 | 0.91 / 0.91 | 0.82 / 0.90 |

- Re-weighting never moves hard-tier off 0.50 → the *rule* is the issue.
- The retrieval-tuned static model reaches the ceiling but is unusable
  globally (help 0.28).
- Gating recovers both. **T=3 is the no-regression knee** (help stays
  1.00/1.00); higher T squeezes a little more corpus_b at the cost of
  help. The exact T is corpus-relative and re-derived per corpus (§4).

## 3. Mechanism — confidence-gated fusion

```
hits = keyword.retrieve(q, corpus, k)
if hits and hits[0].score >= T:      # keyword is confident
    return hits                       # authoritative — don't dilute it
return embedding.retrieve(q, corpus, k)   # rescue the paraphrase
```

Prototype (`GatedRetriever` in `gated_fusion.py`) is ~15 lines.
Candidate refinements for scoping:

- **Hard switch vs blend.** The prototype is a hard switch (keyword
  *or* embedding). Alternative: when below T, *fuse* keyword+embedding
  by RRF rather than going embedding-only — may protect medium queries.
  Measure both at M2.
- **Gate on top-1 score vs top1−top2 gap.** The abstention probe found
  the gap separates legit/neg even better than raw top-1. The gate could
  key on either; pick by measurement (§4) and keep it consistent with
  the abstention signal.
- **k-aware gating.** Gate per-rank (rescue only ranks the keyword leg
  left empty) instead of all-or-nothing. Lower priority.

## 4. Threshold T — corpus-relative, same lesson as `min_score`

T is an absolute keyword score, so (exactly like the abstention
`min_score`) it must be calibrated per corpus, not hardcoded. The
calibration is the same sweep machinery as
`--calibrate-abstention`: find the T that maximizes hard/paraphrase
rescue subject to **zero** regression on the corpus's own legit set.

**This is the unification with [`safe-abstention-defaults`](../safe-abstention-defaults/):**
one keyword-confidence threshold drives both decisions. Below T:
embedding-rescue if the embedding leg is strong, abstain if neither leg
is. The two specs share one calibration tool and one threshold per
corpus. They MUST NOT ship two independent thresholds.

## 5. Model choice

- **`minishlab/potion-retrieval-32M`** — retrieval-tuned model2vec
  distillation. Torch-free, static, ms-encode. Heavier than the current
  `potion-base-8M` default (bigger download/RSS) but same dependency
  class. This is the measured winner.
- Keep the model **injectable / configurable** (as `EmbeddingRetriever`
  already is) so the footprint is a user choice and tests stay
  download-free.
- The current `potion-base-8M` did **not** reach the ceiling even
  embedding-only — so the model upgrade is load-bearing, not cosmetic.

## 6. Recommended shape (for ratification at `/spec`)

1. Opt-in `GatedRetriever` (or a `gate=` mode on `HybridRetriever`)
   under `[embeddings]`, defaulting to `potion-retrieval-32M`.
2. Threshold calibrated per corpus via the **shared** abstention/gate
   calibration tool (designed jointly with `safe-abstention-defaults`).
3. Keyword-only remains the base-install default. No default flip.
4. **Gated on a ≥30-query hard-set validation first** (R1) — the n=4
   hard tier is too thin to build on.

## 7. What this spec does NOT decide

- Whether a torch model is ever needed — only revisited if static
  fails the ≥30-query validation.
- The exact T (re-derived per corpus at calibration time).
- Hard-switch vs below-T-RRF-blend — decided by M2 measurement.
- Whether gating becomes the *default* for arbitrary corpora or stays a
  documented opt-in — decided after validation.

## 8. Open questions for scoping

1. Hard switch vs below-T-RRF-blend?
2. Gate on top-1 score or top1−top2 gap?
3. New `GatedRetriever` class vs a `gate=`/`mode=` option on
   `HybridRetriever`? (API surface → freeze relevance.)
4. One shared calibration tool with `safe-abstention-defaults`, or two
   that read a common threshold? (R-consistency.)
5. Default model: `potion-retrieval-32M` vs an even larger static
   model — footprint vs quality, decided by the ≥30-query run.
