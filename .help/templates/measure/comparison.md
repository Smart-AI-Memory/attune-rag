---
type: comparison
name: measure-comparison
feature: measure
depth: comparison
generated_at: 2026-06-10T06:09:41.203297+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Comparison: Measure retriever tiers

## Context

`attune_rag.measure_corpus` scores retrieval quality — P@1 and R@3 — against a golden-query YAML file. Its most practical use as a comparison tool is evaluating the three retriever tiers (`keyword`, `hybrid`, `transformer`) against the same corpus and query set so you can make an evidence-based decision about whether the accuracy gains justify a heavier dependency.

The `measure()` function accepts a `retriever` parameter and returns a `MeasureResult` dataclass. Run it once per tier, then compare `p1`, `r3`, and (if you supply `paraphrased_path`) `paraphrased_p1` / `paraphrased_r3` across the three results.

---

## Retriever tier comparison

| | `keyword` | `hybrid` | `transformer` |
|---|---|---|---|
| **P@1 / R@3 baseline** | Establishes your floor — no extra dependencies | Typically improves over keyword; requires hybrid extra | Highest ceiling; requires transformer extra |
| **Paraphrase recall (`paraphrased_p1` / `paraphrased_r3`)** | Often weakest — exact-match bias | Moderate improvement | Largest lift — semantic matching handles lexical variation |
| **`candidate_multiplier` impact** | Meaningful — more candidates surface more keyword matches | Moderate | Less critical — dense retrieval already ranks semantically |
| **Install cost** | None — default tier | Medium extra | Heavy extra |
| **Exit behavior when extra is missing** | N/A — always available | Exits 2 with install hint | Exits 2 with install hint |
| **`rerank=True` compatibility** | Supported | Supported | Supported |

### How to read the tradeoff

- **Keyword → Hybrid**: Run both with the same `queries_path` and compare `r3`. If the lift is small (e.g., < 0.05), the hybrid extra may not be worth the install weight for your corpus.
- **Hybrid → Transformer**: Check `paraphrased_r3` specifically. The transformer tier's advantage is most visible on paraphrased queries; if you have no `paraphrased_path`, the improvement can look modest even when semantic matching is genuinely better.
- **`rerank=True`**: Available across all tiers via `candidate_multiplier`. It fetches `candidate_multiplier × k` candidates before reranking, so it increases latency — measure with and without to quantify the accuracy/speed tradeoff.

---

## Corpus source comparison

You can also use `measure()` to compare your own corpus against the bundled one. The two options are mutually exclusive — passing both raises a `ValueError`.

| | `corpus_path=<path>` | `bundled=True` |
|---|---|---|
| **Use case** | Score your production or custom corpus | Baseline / smoke-test against the shipped corpus |
| **`corpus_label` in output** | Reflects your path | Reflects the bundled corpus label |
| **`queries_sha` / `paraphrased_sha`** | Tracks your query file versions | Tracks bundled query versions |
| **When to prefer** | You are tuning or validating a real deployment | You want a known-good reference point |

---

## Output format comparison

`MeasureResult` offers two serialization methods:

| | `report_markdown()` | `to_json()` |
|---|---|---|
| **Best for** | Human review, pull-request comments, CI artifacts | Programmatic diffing, storing results, feeding dashboards |
| **Includes `per_difficulty_breakdown`** | Yes — rendered as a table | Yes — as a nested dict |
| **Includes `per_query_table`** | Yes | Yes |
| **Watermark gating** | Use `watermark_failures()` separately to gate CI | Parse the JSON fields `p1` and `r3` directly |

---

## Use `measure()` when…

| Situation | Recommendation |
|---|---|
| You want to decide whether to install the hybrid or transformer extra | Run `measure()` with `retriever='keyword'` first to establish a baseline, then re-run with `retriever='hybrid'` or `retriever='transformer'` and compare `p1` and `r3`. |
| You have paraphrased queries and want to measure recall robustness | Supply `paraphrased_path` — the transformer tier almost always shows the largest `paraphrased_r3` lift. |
| You need a CI quality gate | Call `watermark_failures(p1_floor=..., r3_floor=...)` on the returned `MeasureResult`; a non-empty list means the corpus failed the gate. |
| You want a quick sanity check against a known-good baseline | Use `bundled=True` — no corpus path needed, exits cleanly on success. |
| You are comparing corpus versions (e.g., before/after a content update) | Run `measure()` with `corpus_path=` for each version and diff the `p1`, `r3`, and `per_difficulty_breakdown` fields. |

**Skip `measure()` if** you only need to retrieve documents at runtime — `measure()` is a scoring harness, not a retrieval interface. If you need to call the retriever directly without scoring against a query set, use the retrieval layer without the measurement harness.

---

**Tags:** `measure`, `corpus`, `quality`, `watermark`, `retriever-tiers`, `p-at-1`, `r-at-3`
