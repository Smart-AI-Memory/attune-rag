---
type: reference
name: benchmark-reference
feature: benchmark
depth: reference
generated_at: 2026-05-20T03:30:01.578655+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark reference

Run retrieval and optional faithfulness benchmarks, gating CI on configurable thresholds. Supports custom query files and optional faithfulness scoring via `--with-faithfulness`.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `main` | `argv: list[str] \| None = None` | `int` | Runs the benchmark suite and returns `0` on success. |

## Source files

- `src/attune_rag/benchmark.py`

## Tags

`benchmark`, `ci`, `precision`, `recall`, `quality`
