---
type: concept
feature: benchmark
depth: concept
generated_at: 2026-04-23T03:36:31.981734+00:00
source_hash: efe3184170bbd1e763131bf4307b2835dc8fb12752af2f0f8b5cb67b4d27ad03
status: generated
---

# Benchmark

## What it is

The benchmark module is a quality gate system that measures retrieval accuracy and optionally evaluates answer faithfulness against ground truth data.

## How it works

The benchmark runner evaluates your RAG system on three dimensions:

- **Precision** — how many retrieved documents are relevant
- **Recall** — how many relevant documents were retrieved
- **Faithfulness** — how well generated answers align with source material (optional)

You configure pass/fail thresholds for each metric. If any threshold isn't met, the benchmark fails and can block CI deployment.

The runner accepts custom query files containing test questions and expected results. When you enable faithfulness scoring with `--with-faithfulness`, the system evaluates not just retrieval quality but also whether generated answers stay true to the retrieved content.

## Entry point

The `main()` function in `src/attune_rag/benchmark.py` orchestrates the entire benchmark process, returning 0 on success or a non-zero exit code when thresholds aren't met.

## Integration points

Other systems interact with the benchmark through:

- **CI pipelines** — call `main()` to gate deployments on quality metrics
- **Custom query sets** — provide domain-specific test cases through file input
- **Threshold configuration** — set precision/recall/faithfulness requirements per environment
