---
type: note
name: measure-note
feature: measure
depth: note
generated_at: 2026-06-10T06:09:41.200922+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Note: measure

## Context

`attune_rag.measure_corpus` scores retrieval quality against a golden-query YAML file. The two public names it exports are `measure` and `MeasureResult`.

Calling `measure()` requires exactly one of `corpus_path=` or `bundled=True`; passing neither or both raises `ValueError`. The function returns a `MeasureResult` dataclass whose primary metrics are `p1` (Precision@1) and `r3` (Recall@3), each computed over `n` queries.

When you supply `paraphrased_path=`, the harness runs a second evaluation pass and populates `paraphrased_p1`, `paraphrased_r3`, and `paraphrased_n` alongside the canonical metrics, letting you compare paraphrase-recall lift without a separate invocation.

## Metrics and reporting

`MeasureResult` carries three output methods:

- `report_markdown()` — renders a human-readable Markdown report; accepts an optional `frozen_timestamp` to pin the datestamp in CI output.
- `to_json()` — serializes the full result, including `per_query_table`, `paraphrased_per_query`, and `per_difficulty_breakdown`.
- `watermark_failures()` — returns a list of metric names that fall below the supplied `p1_floor` and/or `r3_floor` thresholds. An empty list means all gates passed.

The `harness_version` field (currently `'0.1.0'`) is stamped into every `MeasureResult` so that stored JSON reports remain traceable to the harness that produced them.

## Retriever tiers

The `retriever` parameter (default `'keyword'`) is recorded on `MeasureResult.retriever`. If the selected tier requires an optional dependency that is not installed, `measure()` exits with code `2` and prints an install hint rather than raising an unhandled exception.

**Tags:** `measure`, `corpus`, `quality`, `watermark`, `retriever-tiers`, `p-at-1`, `r-at-3`
