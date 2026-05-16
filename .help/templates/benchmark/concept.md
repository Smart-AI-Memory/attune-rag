---
feature: benchmark
depth: concept
generated_at: 2026-05-16T08:44:05.206974+00:00
source_hash: 3ecf192b8459343df49cfb4a43d472af419ebfe935b46e8c0f34d5635079e14d
status: generated
---

# Benchmark

## How it works

Precision/recall/faithfulness benchmark runner — gates CI on configurable thresholds; supports custom query files and optional faithfulness scoring via --with-faithfulness.

The main entry points are:

- **`main()`** — core function

## What connects to it

This feature relates to: benchmark, ci, precision, recall, quality.

Other parts of the codebase call into
benchmark through these functions:

| Function | Purpose | File |
|----------|---------|------|
| `main()` | — | `src/attune_rag/benchmark.py` |
