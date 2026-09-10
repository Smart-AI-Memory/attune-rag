---
type: tip
name: benchmark-tip
feature: benchmark
depth: tip
generated_at: 2026-06-10T06:07:59.727578+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Tip: working effectively with benchmark

Run `attune-rag-benchmark` against a representative query file before merging retriever changes, not after.

**Why:** The benchmark runner exits with a non-zero code when configured thresholds are not met, so catching a regression in CI is far cheaper than diagnosing degraded retrieval quality in production.

**How:** Pass your query file and the retriever tier you changed. For example, if you modified the hybrid tier, target it directly with `--retriever hybrid`. Keyword needs no retriever extra. Install `[embeddings]` before measuring hybrid, which can otherwise fall back to keyword-only. Transformer requires `[transformers]`; a missing dependency exits `2` with an install hint.

Use `--calibrate-abstention` with legitimate and negative query sets to obtain a keyword-threshold recommendation; it does not apply the threshold or complete either benchmark stage. Add `--with-faithfulness` only when you need generation and judge calls.

Save `--json benchmark.json` so completed measurements survive a later failure. Distinguish `1` (regression), `2` (local or invalid-data failure), and `3` (classified transient primary-provider failure). A CLI pass uses its own cutoffs; repository CI separately checks the locked thresholds and query-file hash.

**Tradeoff:** Running all three retriever tiers (`keyword`, `hybrid`, `transformer`) gives you the most complete picture but takes longer. In a time-constrained CI job, target only the tier your change affects.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
