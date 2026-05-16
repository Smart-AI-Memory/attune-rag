---
type: concept
name: benchmark-concept
feature: benchmark
depth: concept
generated_at: 2026-05-15T18:40:20.588450+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark

The benchmark feature is a retrieval and faithfulness quality gate that measures how well the RAG pipeline performs and fails CI when results fall below configurable thresholds.

## What it measures

The benchmark runner evaluates two dimensions of RAG output:

- **Retrieval quality** — precision and recall scores that reflect whether the right documents are being surfaced for a given query.
- **Faithfulness** — an optional check (enabled with `--with-faithfulness`) that scores whether generated answers stay grounded in the retrieved content.

You can supply a custom query file to target specific domains or edge cases rather than relying on defaults.

## How the pieces fit together

At the center is `main()` in `src/attune_rag/benchmark.py`. When called, it runs the configured benchmark suite, computes scores, and returns an exit code — `0` for pass, non-zero for fail. That exit code is what CI uses to gate a build: if precision, recall, or faithfulness drops below a threshold, the pipeline stops.

The flow looks like this:

1. `main()` loads queries (default or from a custom file).
2. It runs retrieval and scores precision and recall.
3. If `--with-faithfulness` is set, it also scores answer grounding.
4. It compares each score against its configured threshold.
5. It exits `0` if all thresholds pass, non-zero otherwise.

## When it matters

Run the benchmark when you want confidence that a change to the retrieval pipeline — new chunking strategy, updated embeddings, adjusted ranking — hasn't degraded quality. Because `main()` returns a standard exit code, plugging it into CI is straightforward: a regression fails the build before it reaches production.
