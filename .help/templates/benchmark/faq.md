---
type: faq
name: benchmark-faq
feature: benchmark
depth: faq
generated_at: 2026-06-10T06:07:59.722303+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Benchmark FAQ

## What does the benchmark feature do?

It runs precision, recall, and optional faithfulness benchmarks against your RAG pipeline and exits with a non-zero code when results fall below your configured thresholds, so you can gate CI on retrieval quality.

## How do I run it?

Run the `attune-rag-benchmark` console script from your terminal. It calls `attune_rag.benchmark.main()` and returns `0` on success.

## Which retrieval tiers can I benchmark?

Select one tier per run: `--retriever keyword`, `--retriever hybrid`, or `--retriever transformer`. Keyword needs no retriever extra. Hybrid uses `[embeddings]` and can fall back to keyword-only if its embedding leg is unavailable. Transformer needs `[transformers]`; a missing dependency exits `2` with an install hint.

## How do I add faithfulness scoring?

Pass `--with-faithfulness`. It is off by default because it generates and judges an answer for each query. Answer generation requires `ANTHROPIC_API_KEY` and uses API tokens even when the judge uses a subscription route.

## What is abstention-threshold calibration and when do I need it?

Use `--calibrate-abstention --negatives negative_queries.yaml` to recommend an absolute keyword-score threshold from legitimate and negative queries. It does not apply the recommendation. Successful JSON output marks both benchmark stages `skipped` with reason `calibration_only`, and includes `calibration` results.

## Can I benchmark against my own queries?

Yes. Pass a custom query file to supply your own query set instead of the built-in one. See the CLI help (`attune-rag-benchmark --help`) for the exact flag syntax.

## What exit codes does the benchmark command return?

| Code | Meaning |
|------|---------|
| `0` | Requested mode completed successfully; normal benchmark CLI cutoffs passed, or calibration completed |
| `1` | A completed measurement fell below a CLI cutoff |
| `2` | Local, credential, schema, dependency, or invalid-data failure |
| `3` | Classified transient failure during the primary provider pass |

## Why can a local pass still fail repository CI?

The CLI defaults are precision@1 ≥ `0.70` and optional primary faithfulness ≥ `0.85`; it does not gate recall. Repository CI checks `docs/specs/release-quality-baseline/thresholds.json` separately: precision@1 ≥ `0.975`, recall@3 ≥ `1.0`, and primary faithfulness ≥ `0.9698` when enabled. It also verifies the query-file hash.

## What does `--json` retain after failure?

Completed retrieval and primary faithfulness are checkpointed before later work. Inspect `outcomes.retrieval`, `outcomes.faithfulness`, and `outcomes.reason`; controlled local errors may add `outcomes.detail`. A primary provider outage leaves retrieval available, while a failed optional comparison preserves the completed primary result.

## Where is the source?

`src/attune_rag/benchmark.py` — the public entry point is `attune_rag.benchmark.main(argv: list[str] | None = None) -> int`.

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
