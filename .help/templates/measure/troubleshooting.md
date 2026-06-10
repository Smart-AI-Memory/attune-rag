---
type: troubleshooting
name: measure-troubleshooting
feature: measure
depth: troubleshooting
generated_at: 2026-06-10T06:09:41.191870+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Troubleshoot measure

## Before you start

`attune_rag.measure_corpus` scores P@1 and R@3 against a golden-query YAML file. You must supply either `corpus_path=` (a path to your own corpus) or `bundled=True` (the built-in corpus) — passing both, or neither, raises `ValueError`. The `retriever` parameter accepts `'keyword'` (default), `'hybrid'`, or `'transformer'`; the non-keyword tiers require optional extras, and `measure()` exits with code 2 and an install hint if the required extra is missing. Results are returned as a `MeasureResult` and can be serialized with `.report_markdown()` or `.to_json()`. Pass thresholds to `.watermark_failures()` to gate on `p1` or `r3`.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `ValueError: measure() requires exactly one of corpus_path= or bundled=True.` | You passed both `corpus_path=` and `bundled=True`, or neither. Pass exactly one. |
| Exit code 2 with an install hint | The selected `retriever` tier requires an extra that is not installed. Run the printed `pip install` command. |
| `p1` or `r3` is lower than expected | `watermark_failures()` returns the failing metric names — call it with your `p1_floor` and `r3_floor` thresholds to confirm which metric fell short. |
| `paraphrased_p1` / `paraphrased_r3` are `None` | `paraphrased_path=` was not passed to `measure()`. Paraphrase metrics are only populated when a paraphrased query file is provided. |
| `per_difficulty_breakdown` is empty | The query YAML contains no difficulty annotations, so the breakdown dict defaults to `{}`. |
| Results differ across runs with the same inputs | Check whether `rerank=True` is set and whether the `retriever` value changed between runs. Both are recorded on `MeasureResult.rerank` and `MeasureResult.retriever`. |
| `harness_version` in the JSON report is unexpected | The value is fixed at `'0.1.0'` for this release. A mismatch between reports indicates they were produced by different installed versions of `attune_rag`. |

## Diagnosis steps

1. **Confirm the exact call.** Print or log the arguments you are passing to `measure()` — specifically `corpus_path`, `bundled`, `queries_path`, `retriever`, `rerank`, and `candidate_multiplier`. Most failures trace back to a wrong or missing argument.

2. **Run the minimal call.** Reduce to the required arguments only and confirm whether the failure persists:
   ```python
   from attune_rag.measure_corpus import measure
   result = measure(bundled=True, queries_path="your_queries.yaml")
   print(result.p1, result.r3, result.n)
   ```
   If this succeeds, add back optional arguments one at a time to isolate the culprit.

3. **Check watermark failures explicitly.** If scores are lower than expected, call:
   ```python
   failures = result.watermark_failures(p1_floor=0.8, r3_floor=0.9)
   print(failures)
   ```
   An empty list means both thresholds passed. A non-empty list names the metric(s) that fell short.

4. **Inspect the per-query breakdown.** Iterate `result.per_query_table` to find which individual queries are failing — this narrows whether the problem is corpus-wide or limited to specific queries.

5. **Compare retriever tiers.** If you suspect the retriever is the issue, run `measure()` twice with `retriever='keyword'` and your target retriever, keeping all other arguments identical. Compare `p1` and `r3` across the two `MeasureResult` objects.

6. **Run the related tests.** Execute `pytest -k "measure" -v` to confirm whether the failure is reproducible in the test suite. A passing test that exercises your failing path is a useful fixture baseline.

## Common fixes

- **Supply exactly one corpus source.** Either pass `corpus_path="/path/to/corpus"` or set `bundled=True`, never both:
  ```python
  # correct — own corpus
  measure(corpus_path="/path/to/corpus", queries_path="queries.yaml")

  # correct — bundled corpus
  measure(bundled=True, queries_path="queries.yaml")
  ```

- **Install the retriever extra.** If you see an exit-code-2 error mentioning a missing extra, run the `pip install` command printed in the error message. The `'keyword'` retriever requires no extras and always works.

- **Provide a paraphrased query file to get paraphrase metrics.** `paraphrased_p1` and `paraphrased_r3` are `None` unless you pass `paraphrased_path=`:
  ```python
  measure(bundled=True, queries_path="queries.yaml", paraphrased_path="paraphrased.yaml")
  ```

- **Pin or reconcile the installed version.** If two `MeasureResult` JSON reports show different `harness_version` values, the reports came from different installs. Run `pip show attune-rag` on each environment and align them.

- **Check `queries_sha` and `paraphrased_sha` for drift.** `MeasureResult` records the SHA of both query files. If results change unexpectedly between runs, compare these fields to confirm the query files themselves have not changed.

## Source files

- `src/attune_rag/measure_corpus.py`

**Tags:** `measure`, `corpus`, `quality`, `watermark`, `retriever-tiers`, `p-at-1`, `r-at-3`
