# Diagnostic-2 — QueryExpander robustness

corpus: attune-help  version: 0.11.0
baseline queries: 40; paraphrased: 80
unique queries expanded: 120
cache: docs/specs/embedding-retriever/diagnostic-2.cache.json

## Baseline (`queries.yaml`)

### Keyword + QueryExpander

| difficulty | n | P@1 | R@3 |
|---|---|---|---|
| easy | 10 | 90.00% | 100.00% |
| medium | 27 | 85.19% | 96.30% |
| hard | 3 | 100.00% | 100.00% |
| **overall** | **40** | **87.50%** | **97.50%** |

**Compare to D1 keyword-only baseline:** P@1 97.50%, R@3 100.00%

## Paraphrased (`queries_paraphrased.yaml`)

### Keyword + QueryExpander

| difficulty | n | P@1 | R@3 |
|---|---|---|---|
| easy | 20 | 50.00% | 80.00% |
| medium | 54 | 50.00% | 79.63% |
| hard | 6 | 66.67% | 83.33% |
| **overall** | **80** | **51.25%** | **80.00%** |

**Compare to D1 keyword-only paraphrased:** P@1 11.25%, R@3 28.75%

Δ P@1 vs keyword-only on paraphrased: +40.00%
Δ R@3 vs keyword-only on paraphrased: +51.25%

## Verdict

**STRONG** — QueryExpander alone closes most of the paraphrase gap. Combined with the alias-expansion lever (D3), embeddings are very likely unnecessary. Spec defers; follow-up is to recommend QueryExpander in docs and consider making it default for users with `[claude]` installed.

Summary stats written to: docs/specs/embedding-retriever/diagnostic-2.summary.json

---

**Run metadata:** 120 Haiku calls, 160.9s wall time, completed 2026-05-21.
