---
type: faq
name: measure-faq
feature: measure
depth: faq
generated_at: 2026-06-10T06:09:41.194237+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Measure FAQ

## What does `measure` do?

It scores your corpus against a golden-query file and returns P@1 and R@3 metrics as a `MeasureResult`. You can run it programmatically via `measure()` in `attune_rag.measure_corpus` or from the command line with the `attune-rag-measure` console script.

## When should I use it?

Use `measure` when you want to evaluate retrieval quality against a known set of queries — for example, before and after switching `retriever` values, or to verify that paraphrase recall improves before adding a heavier dependency.

## What retriever options are available?

Pass `retriever='keyword'` (the default), `'hybrid'`, or `'transformer'` to `measure()`. Each tier compares differently; if a tier's required extra is not installed, the call exits with an install hint.

## How do I point `measure` at my own corpus?

Pass `corpus_path=` with a path to your corpus directory. Alternatively, pass `bundled=True` to use the bundled corpus. You must supply exactly one of these — passing both or neither raises a `ValueError`.

## What is `candidate_multiplier` for?

It controls how many candidates the retriever fetches before re-ranking. The default is `3`. Increase it if you're using `rerank=True` and want a larger pool of candidates to re-rank against.

## How do I measure paraphrase recall?

Pass `paraphrased_path=` with a path to your paraphrased queries file. The returned `MeasureResult` will populate `paraphrased_p1`, `paraphrased_r3`, and `paraphrased_n` alongside the primary metrics.

## What does `MeasureResult` give me?

The `MeasureResult` dataclass contains:

- `p1` and `r3` — overall Precision@1 and Recall@3 scores
- `n` — number of queries evaluated
- `paraphrased_p1`, `paraphrased_r3`, `paraphrased_n` — same metrics for the paraphrased query set (if you supplied `paraphrased_path`)
- `per_query_table` — per-query scores as a tuple of `QueryScore` objects
- `per_difficulty_breakdown` — scores broken down by difficulty tier
- `retriever`, `rerank`, `harness_version`, and path/SHA fields for reproducibility

## How do I check whether my results meet a quality gate?

Call `result.watermark_failures(p1_floor=..., r3_floor=...)`. It returns a list of failure strings — an empty list means all thresholds passed.

## How do I get a report I can share?

- `result.report_markdown()` returns a formatted Markdown report. Pass `frozen_timestamp=` to pin the timestamp for reproducible output.
- `result.to_json()` returns the full result as a JSON string.

## Where is the source?

`attune_rag.measure_corpus` — importable as `from attune_rag.measure_corpus import measure, MeasureResult`.

**Tags:** `measure`, `corpus`, `quality`, `watermark`, `retriever-tiers`, `p-at-1`, `r-at-3`
