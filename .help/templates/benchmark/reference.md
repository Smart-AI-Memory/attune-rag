---
type: reference
name: benchmark-reference
feature: benchmark
depth: reference
generated_at: 2026-06-10T06:07:59.704964+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark reference

Precision/recall/faithfulness benchmark runner, installed as the `attune-rag-benchmark` console script. Gates CI on configurable thresholds; `--retriever {keyword,hybrid,transformer}` benchmarks each retrieval tier (exits `2` with an install hint when the tier's extra is missing); supports custom query files, abstention-threshold calibration via `--calibrate-abstention`, and optional faithfulness scoring via `--with-faithfulness`.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `main` | `argv: list[str] \| None = None` | `int` | Runs the benchmark suite and returns an exit code. Returns `0` on success. |

## Source files

- `src/attune_rag/benchmark.py`

## Tags

`benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
