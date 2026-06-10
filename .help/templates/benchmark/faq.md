---
type: faq
name: benchmark-faq
feature: benchmark
depth: faq
generated_at: 2026-06-10T06:07:59.722303+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark FAQ

## What does the benchmark feature do?

It runs precision, recall, and optional faithfulness benchmarks against your RAG pipeline and exits with a non-zero code when results fall below your configured thresholds, so you can gate CI on retrieval quality.

## How do I run it?

Run the `attune-rag-benchmark` console script from your terminal. It calls `attune_rag.benchmark.main()` and returns `0` on success.

## Which retrieval tiers can I benchmark?

Pass `--retriever keyword`, `--retriever hybrid`, or `--retriever transformer`. If the required extra for a tier isn't installed, the command exits with code `2` and prints an install hint.

## How do I add faithfulness scoring?

Pass `--with-faithfulness`. Faithfulness scoring is off by default because it requires an additional model call.

## What is abstention-threshold calibration and when do I need it?

Use `--calibrate-abstention` when you want the benchmark to determine the confidence threshold below which your retriever should decline to answer rather than return a low-quality result.

## Can I benchmark against my own queries?

Yes. Pass a custom query file to supply your own query set instead of the built-in one. See the CLI help (`attune-rag-benchmark --help`) for the exact flag syntax.

## What exit codes does the benchmark command return?

| Code | Meaning |
|------|---------|
| `0` | All thresholds passed |
| `2` | A required extra for the requested retriever tier is not installed |

Any other non-zero exit indicates a threshold failure or runtime error.

## Where is the source?

`src/attune_rag/benchmark.py` — the public entry point is `attune_rag.benchmark.main(argv: list[str] | None = None) -> int`.

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
