---
type: quickstart
name: measure-quickstart
feature: measure
depth: quickstart
generated_at: 2026-06-10T06:09:41.196620+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Quickstart: Measure Retrieval Quality

Score your corpus against a golden-query file and see P@1 and R@3 in your terminal.

```python
from pathlib import Path
from attune_rag.measure_corpus import measure

result = measure(
    corpus_path="my_corpus/",
    queries_path="queries.yaml",
)
print(result.p1, result.r3)
```

Expected output:

```
0.82 0.91
```

## Step 1: Install and import

Install `attune-rag` so that `attune_rag.measure_corpus` is available, then import the two public names you need:

```python
from attune_rag.measure_corpus import measure, MeasureResult
```

## Step 2: Run `measure()` against your corpus

Call `measure()` with either `corpus_path=` pointing at your local corpus directory or `bundled=True` to use the built-in corpus — you must provide exactly one of those two, or the call raises `ValueError`. Pass `queries_path=` with your golden-query YAML:

```python
result = measure(
    corpus_path="my_corpus/",
    queries_path="queries.yaml",
)
```

## Step 3: Read the scores

`measure()` returns a `MeasureResult`. The two headline fields are `p1` (precision at 1) and `r3` (recall at 3), and `n` tells you how many queries were evaluated:

```python
print(f"P@1={result.p1:.2f}  R@3={result.r3:.2f}  n={result.n}")
```

Expected output:

```
P@1=0.82  R@3=0.91  n=120
```

## Step 4: Export a report

Call `report_markdown()` on the result to get a formatted report you can save or share, or call `to_json()` if you need machine-readable output:

```python
print(result.report_markdown())
```

Expected output (truncated):

```
## Retrieval quality report
| metric | value |
|--------|-------|
| P@1    | 0.82  |
| R@3    | 0.91  |
...
```

Next: check whether your scores clear your quality gates by calling `result.watermark_failures(p1_floor=0.80, r3_floor=0.90)` — it returns an empty list when both thresholds pass.
