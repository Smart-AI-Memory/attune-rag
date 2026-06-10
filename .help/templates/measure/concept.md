---
type: concept
name: measure-concept
feature: measure
depth: concept
generated_at: 2026-06-10T06:09:41.169476+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Measure

`measure` is a retrieval-quality harness that scores how well a corpus answers a set of golden queries, returning a `MeasureResult` with precision and recall metrics you can gate in CI or explore interactively.

## Core concepts

**Precision at 1 (P@1)** and **Recall at 3 (R@3)** are the two headline metrics. P@1 asks whether the top-ranked result is correct; R@3 asks whether the correct answer appears anywhere in the top three. Both are expressed as floats between 0 and 1 and stored in `MeasureResult.p1` and `MeasureResult.r3`.

**A corpus + query set** is the unit of measurement. You supply a corpus (via `corpus_path` or `bundled=True`) and a queries file (`queries_path`). The `measure()` function runs every query against the corpus and accumulates scores into a single `MeasureResult`. Passing `paraphrased_path` runs a second pass on paraphrased versions of the same queries, which populates the parallel `paraphrased_p1`, `paraphrased_r3`, and `paraphrased_n` fields so you can see recall lift from paraphrase expansion without re-indexing.

**Retriever tiers** control how candidates are fetched. The `retriever` parameter (default `'keyword'`) selects the retrieval strategy. Because heavier tiers require optional extras, the harness exits early with an install hint when the required extra is missing, so you can compare tiers incrementally.

**`candidate_multiplier`** scales the number of candidates fetched before reranking. When `rerank=True`, the harness fetches `candidate_multiplier × k` results and reranks them; this is reflected in `MeasureResult.rerank`.

## The MeasureResult object

`MeasureResult` is a dataclass that captures everything needed to reproduce and report a measurement run:

| Field | Type | What it holds |
|---|---|---|
| `p1` | `float` | Precision at 1 across all queries |
| `r3` | `float` | Recall at 3 across all queries |
| `n` | `int` | Number of queries scored |
| `paraphrased_p1` | `float \| None` | P@1 on paraphrased queries, if supplied |
| `paraphrased_r3` | `float \| None` | R@3 on paraphrased queries, if supplied |
| `per_query_table` | `tuple[QueryScore, ...]` | Per-query breakdown for the primary query set |
| `per_difficulty_breakdown` | `dict[str, dict[str, float]]` | Scores grouped by difficulty label |
| `queries_sha` | `str` | SHA of the queries file, for reproducibility |
| `retriever` | `str` | Retriever tier used in this run |
| `harness_version` | `str` | Version of the measurement harness (`0.1.0`) |

Once you have a `MeasureResult`, three methods let you act on it:

- `watermark_failures(p1_floor=..., r3_floor=...)` — returns a list of strings describing any metric that fell below the given thresholds. An empty list means the run passed all gates.
- `report_markdown()` — renders a human-readable Markdown report suitable for pull-request comments or documentation.
- `to_json()` — serializes the full result for storage or downstream tooling.

## How the pieces fit together

```
corpus + queries
      │
      ▼
  measure()          ← corpus_path= / bundled=, queries_path=,
      │                retriever=, rerank=, paraphrased_path=
      ▼
 MeasureResult
  ├── .p1 / .r3                  (headline metrics)
  ├── .per_query_table           (drill down to individual queries)
  ├── .per_difficulty_breakdown  (slice by difficulty label)
  ├── .watermark_failures()      (CI gate)
  ├── .report_markdown()         (human report)
  └── .to_json()                 (machine output)
```

You call `measure()` once per run; it returns a single `MeasureResult` that you can inspect interactively, serialize to JSON for trend tracking, or pass through `watermark_failures()` to enforce quality gates in CI.

## When measurement matters

- **Before switching retriever tiers** — run `measure()` with `retriever='keyword'` and again with a heavier tier to quantify the recall improvement before adding the dependency.
- **After changing chunking or indexing** — a drop in `p1` or `r3` signals a regression even when manual spot-checks look fine.
- **In CI** — call `watermark_failures(p1_floor=0.8, r3_floor=0.9)` and fail the build if the returned list is non-empty.
- **With paraphrased queries** — pass `paraphrased_path` to measure whether your retriever generalises beyond the exact wording in the golden query set.
