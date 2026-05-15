---
feature: benchmark
depth: concept
generated_at: 2026-05-15T13:03:36.549184+00:00
source_hash: fa744d71791100502f1e27d84431e0d8aa1381376327b87453c04f3f56c22384
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
