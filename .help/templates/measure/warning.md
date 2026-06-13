---
type: warning
name: measure-warning
feature: measure
depth: warning
generated_at: 2026-06-10T06:09:41.189604+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Measure cautions

## What to watch for

`measure()` scores P@1 and R@3 for a corpus-and-query pair and returns a `MeasureResult`. Before you run it, note these constraints:

- You must supply **exactly one** of `corpus_path=` or `bundled=True`. Passing both, or neither, raises `ValueError: 'measure() requires exactly one of corpus_path= or bundled=True.'`
- `retriever` defaults to `'keyword'`. Switching to a heavier tier (`'hybrid'` or `'transformer'`) without the matching extra installed causes the run to exit with an install hint rather than a score — you will not get a `MeasureResult` back.
- Paraphrase metrics (`paraphrased_p1`, `paraphrased_r3`, `paraphrased_n`) are `None` when `paraphrased_path` is omitted. Code that reads those fields without a `None` check will raise at report time, not at measurement time.

## Risk areas

### `corpus_path` and `bundled` are mutually exclusive

Passing `corpus_path=` alongside `bundled=True` raises a `ValueError`. The error fires inside `measure()`, so any setup work your caller did before the call is wasted. Decide which corpus source applies and pass only that argument.

### `candidate_multiplier` silently widens the candidate pool

`candidate_multiplier` (default `3`) multiplies the number of retrieval candidates before re-ranking. A value that is too low artificially caps recall and produces a `r3` score that looks worse than the retriever actually is. A value that is too high increases latency without improving scores. If you are comparing runs, keep `candidate_multiplier` constant across them; a mismatch makes the numbers incomparable.

### `watermark_failures()` returns an empty list when thresholds are omitted

`MeasureResult.watermark_failures()` accepts `p1_floor` and `r3_floor` as keyword-only arguments, both defaulting to `None`. When you omit both, the method always returns `[]` — no failures, regardless of actual scores. Watermark gating only works when you explicitly pass at least one floor value.

### Paraphrase fields require `paraphrased_path`

`MeasureResult.paraphrased_p1`, `paraphrased_r3`, `paraphrased_n`, and `paraphrased_per_query` are all `None` when `paraphrased_path` is not supplied to `measure()`. If your reporting code or CI gate reads these fields unconditionally, it will encounter `None` values that silently produce misleading comparisons rather than errors.

### `harness_version` is stamped at run time

`MeasureResult.harness_version` is set to the `_HARNESS_VERSION` constant (`'0.1.0'`) at the time `measure()` runs. If you deserialise a stored result via `to_json()` and compare it against a result from a newer version of the library, differing `harness_version` values mean the scoring logic may not be equivalent — do not average or diff scores across versions without verifying compatibility.

## How to avoid problems

1. **Guard the corpus-source argument.** Before calling `measure()`, assert that exactly one of `corpus_path` or `bundled` is set. This surfaces the conflict at call-site configuration time rather than inside the library.

2. **Always pass explicit floor values to `watermark_failures()`.** Calls like `result.watermark_failures(p1_floor=0.7, r3_floor=0.8)` are the only ones that actually gate on quality. A bare `result.watermark_failures()` call is a no-op.

3. **Check paraphrase fields before use.** If `paraphrased_path` is optional in your workflow, treat `paraphrased_p1`, `paraphrased_r3`, and `paraphrased_n` as optional too, and guard every read with a `None` check.

4. **Pin `candidate_multiplier` in benchmarks.** Store the value used alongside `MeasureResult.to_json()` output so that comparisons between runs are meaningful.

5. **Record `harness_version` in long-lived reports.** `MeasureResult.report_markdown()` and `to_json()` both include `harness_version`. When storing results for trend analysis, treat a version change as a potential scoring discontinuity.

## Source files

- `src/attune_rag/measure_corpus.py`

**Tags:** `measure`, `corpus`, `quality`, `watermark`, `retriever-tiers`, `p-at-1`, `r-at-3`
