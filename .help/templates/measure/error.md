---
type: error
name: measure-error
feature: measure
depth: error
generated_at: 2026-06-10T06:09:41.183715+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Measure errors

## Common error signatures

These are the most likely errors you'll encounter when calling `measure()` or running the `attune_rag.measure_corpus` entry point:

- **`ValueError: measure() requires exactly one of corpus_path= or bundled=True.`**
  Raised by `measure()` when you pass both `corpus_path=` and `bundled=True`, or neither. Exactly one must be set.

- **Watermark gate failures** — `MeasureResult.watermark_failures()` returns a non-empty list when your `p1` score falls below `p1_floor` or your `r3` score falls below `r3_floor`. This is not a Python exception, but it signals that retrieval quality did not meet the thresholds you specified.

- **Missing retriever extra (exit code 2)** — When you pass `retriever='hybrid'` or `retriever='transformer'` and the required package is not installed, the harness exits with code 2 and prints an install hint. Your `main()` call returns a non-zero exit code instead of a `MeasureResult`.

- **File or path errors** — Passing an invalid path to `queries_path`, `corpus_path`, `paraphrased_path`, or `extra_aliases_file` produces an `OSError` or `FileNotFoundError` before any scoring begins.

## How to diagnose

1. **Confirm you set exactly one corpus source.** `measure()` requires either `corpus_path=<path>` or `bundled=True` — not both, not neither. A `ValueError` with the message `measure() requires exactly one of corpus_path= or bundled=True.` means this invariant was violated.

2. **Check your `retriever=` value against installed extras.** `measure()` accepts `retriever='keyword'` (default), `'hybrid'`, or `'transformer'`. If you specify a non-default tier and the corresponding package is absent, the call exits with code 2 rather than raising a Python exception. Verify your environment has the required extra installed before switching retriever tiers.

3. **Inspect the `MeasureResult` fields before calling `watermark_failures()`.** If `watermark_failures(p1_floor=..., r3_floor=...)` returns a non-empty list, check `MeasureResult.p1`, `MeasureResult.r3`, `MeasureResult.paraphrased_p1`, and `MeasureResult.paraphrased_r3` directly to see which scores fell short. The `per_query_table` and `per_difficulty_breakdown` fields can help you identify which queries or difficulty bands are dragging the aggregate score down.

4. **Verify all file paths before calling `measure()`.** The `queries_path` argument is required. The `corpus_path`, `paraphrased_path`, and `extra_aliases_file` arguments are optional but must point to readable files when provided. An `OSError` or `FileNotFoundError` at startup means a path was resolved before any retrieval ran.

5. **Check `MeasureResult.harness_version`.** If you are comparing results across runs and scores look inconsistent, confirm that `harness_version` is the same across both `MeasureResult` objects. A version mismatch (`'0.1.0'` vs a newer value) can mean the scoring logic changed between runs.

## Source files

- `src/attune_rag/measure_corpus.py`

**Tags:** `measure`, `corpus`, `quality`, `watermark`, `retriever-tiers`, `p-at-1`, `r-at-3`
