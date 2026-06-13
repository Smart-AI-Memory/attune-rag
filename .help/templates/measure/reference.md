---
type: reference
name: measure-reference
feature: measure
depth: reference
generated_at: 2026-06-10T06:09:41.179574+00:00
source_hash: cf7629e165810184528831fb505d3008f4b57cc175e33b16af8fba2f856fa95f
status: generated
---

# Measure reference

Use `attune_rag.measure_corpus` to score retrieval quality (P@1 and R@3) against a golden-query YAML for a local corpus or the bundled corpus. Pass a `queries_path` to run the primary query set; optionally supply `paraphrased_path` to measure paraphrase-recall lift. Choose a retriever tier with the `retriever` argument and gate CI on watermark thresholds via `MeasureResult.watermark_failures()`. Results are available as a `MeasureResult` dataclass, a Markdown report, or JSON.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `measure` | `*, corpus_path: Path \| str \| None = None, bundled: bool = False, queries_path: Path \| str, paraphrased_path: Path \| str \| None = None, rerank: bool = False, candidate_multiplier: int = 3, extra_aliases_file: Path \| str \| None = None, retriever: str = 'keyword'` | `MeasureResult` | Score a corpus + query set and return a `MeasureResult`. |
| `main` | `argv: list[str] \| None = None` | `int` | CLI entry point; returns `0` on success. |

### Raises

| Function | Raises | Message |
|----------|--------|---------|
| `measure` | `ValueError` | `'measure() requires exactly one of corpus_path= or bundled=True.'` |

## Classes

| Class | Description |
|-------|-------------|
| `MeasureResult` | Result of one `measure()` invocation against a corpus + queries. |

### `MeasureResult` fields

| Field | Type | Default |
|-------|------|---------|
| `p1` | `float` | |
| `r3` | `float` | |
| `n` | `int` | |
| `paraphrased_p1` | `float \| None` | |
| `paraphrased_r3` | `float \| None` | |
| `paraphrased_n` | `int \| None` | |
| `per_query_table` | `tuple[QueryScore, ...]` | |
| `paraphrased_per_query` | `tuple[QueryScore, ...] \| None` | |
| `per_difficulty_breakdown` | `dict[str, dict[str, float]]` | `field(default_factory=dict)` |
| `corpus_label` | `str` | `''` |
| `queries_path` | `str` | `''` |
| `queries_sha` | `str` | `''` |
| `paraphrased_path` | `str \| None` | `None` |
| `paraphrased_sha` | `str \| None` | `None` |
| `rerank` | `bool` | `False` |
| `retriever` | `str` | `'keyword'` |
| `harness_version` | `str` | `_HARNESS_VERSION` |

### `MeasureResult` methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `watermark_failures` | `*, p1_floor: float \| None = None, r3_floor: float \| None = None` | `list[str]` | Return a list of watermark violations given optional P@1 and R@3 floor thresholds. |
| `report_markdown` | `*, frozen_timestamp: str \| None = None` | `str` | Render a Markdown report of the measurement results. |
| `to_json` | | `str` | Serialize the result to a JSON string. |

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `_HARNESS_VERSION` | `str` | `'0.1.0'` |

## Source files

- `src/attune_rag/measure_corpus.py`

## Tags

`measure`, `corpus`, `quality`, `watermark`, `retriever-tiers`, `p-at-1`, `r-at-3`
