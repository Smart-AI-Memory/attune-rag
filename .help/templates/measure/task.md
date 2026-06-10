---
type: task
name: measure-task
feature: measure
depth: task
generated_at: 2026-06-10T06:09:41.174780+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Measure retrieval quality

Use `measure` when you want to score how well your corpus answers a set of golden queries — so you can compare retrieval tiers, validate paraphrase recall, and enforce quality gates before shipping a change.

## Prerequisites

- A queries file in YAML format, passed via `queries_path`
- A corpus available either as a local directory (`corpus_path`) or as the bundled corpus (`bundled=True`)
- `attune_rag` installed with the extras required for the retriever tier you intend to use

## Score a corpus

1. **Import `measure` from `attune_rag.measure_corpus`.**

   ```python
   from attune_rag.measure_corpus import measure
   ```

2. **Call `measure()` with your corpus and queries.**

   Pass exactly one of `corpus_path` or `bundled=True`. Omitting both, or supplying both, raises a `ValueError`.

   ```python
   result = measure(
       corpus_path="path/to/corpus",
       queries_path="path/to/queries.yaml",
   )
   ```

3. **Choose a retriever tier.**

   Set `retriever` to `"keyword"` (default), `"hybrid"`, or `"transformer"`. Each tier affects `p1` and `r3` scores differently. If the chosen tier's extra is not installed, `measure` exits with code `2` and prints an install hint.

   ```python
   result = measure(
       corpus_path="path/to/corpus",
       queries_path="path/to/queries.yaml",
       retriever="hybrid",
   )
   ```

4. **Optionally add paraphrase queries and reranking.**

   Supply `paraphrased_path` to measure recall on paraphrased variants of your queries. Set `rerank=True` to apply a reranking pass. Use `candidate_multiplier` to control how many candidates are retrieved before reranking (default: `3`).

   ```python
   result = measure(
       corpus_path="path/to/corpus",
       queries_path="path/to/queries.yaml",
       paraphrased_path="path/to/paraphrased.yaml",
       rerank=True,
       candidate_multiplier=5,
   )
   ```

5. **Inspect the result.**

   `measure()` returns a `MeasureResult`. Check the primary metrics directly:

   ```python
   print(result.p1)   # precision@1
   print(result.r3)   # recall@3
   print(result.n)    # number of queries scored
   ```

   For paraphrased queries, use `result.paraphrased_p1` and `result.paraphrased_r3` (both `None` when `paraphrased_path` was not supplied).

## Check quality gates

Call `watermark_failures()` to get a list of metric names that fall below your thresholds. An empty list means all gates passed.

```python
failures = result.watermark_failures(p1_floor=0.8, r3_floor=0.9)
if failures:
    print("Quality gates failed:", failures)
```

## Export the results

- **Markdown report** — suitable for logging or pull-request comments:

  ```python
  print(result.report_markdown())
  ```

- **JSON** — suitable for CI artifacts or downstream processing:

  ```python
  print(result.to_json())
  ```

## Verify success

Your run succeeded when:

- `measure()` returns a `MeasureResult` without raising a `ValueError`
- `result.n` matches the number of queries in your queries file
- `result.watermark_failures(p1_floor=..., r3_floor=...)` returns an empty list for your target thresholds
