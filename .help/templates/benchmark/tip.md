---
type: tip
name: benchmark-tip
feature: benchmark
depth: tip
generated_at: 2026-06-10T06:07:59.727578+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Tip: working effectively with benchmark

Run `attune-rag-benchmark` against a representative query file before merging retriever changes, not after.

**Why:** The benchmark runner exits with a non-zero code when configured thresholds are not met, so catching a regression in CI is far cheaper than diagnosing degraded retrieval quality in production.

**How:** Pass your query file and the retriever tier you changed. For example, if you modified the hybrid tier, target it directly with `--retriever hybrid`. If the tier's optional dependency is not installed, the runner exits with code 2 and prints an install hint — resolve that before reading into the results.

Use `--calibrate-abstention` when you adjust abstention logic, and add `--with-faithfulness` only when you need faithfulness scoring — it increases runtime noticeably.

**Tradeoff:** Running all three retriever tiers (`keyword`, `hybrid`, `transformer`) gives you the most complete picture but takes longer. In a time-constrained CI job, target only the tier your change affects.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
