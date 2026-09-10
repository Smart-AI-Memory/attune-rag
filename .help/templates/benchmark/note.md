---
type: note
name: benchmark-note
feature: benchmark
depth: note
generated_at: 2026-06-10T06:07:59.729746+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Note: benchmark

## Context

`attune_rag.benchmark` is a precision/recall/faithfulness benchmark runner installed as the `attune-rag-benchmark` console script. Its public surface is a single entry point, `main()`, which returns `0` on success.

The module is designed to gate CI pipelines on configurable quality thresholds rather than serve as a library. You invoke it directly from the command line or call `main()` programmatically; no class instantiation is required.

## Design decisions

**Retrieval tiers are opt-in.** The `--retriever` flag selects `keyword`, `hybrid`, or `transformer` for one run. Keyword needs no retriever extra. Hybrid uses `[embeddings]` but can fall back to keyword-only; transformer requires `[transformers]` and exits `2` with an install hint when it is missing.

**Faithfulness scoring is off by default.** Pass `--with-faithfulness` to enable it. It generates and judges an answer per query; answer generation uses API tokens even with a subscription-routed judge.

**Abstention threshold calibration is built in.** `--calibrate-abstention` recommends a keyword-score threshold from `--queries` and `--negatives`; it does not apply it. JSON records the calibration and marks both benchmark stages `skipped` (`calibration_only`).

**Results survive later failures.** With `--json`, completed retrieval and primary faithfulness are saved before later work. The `outcomes` object distinguishes completed, skipped, unavailable, and failed stages. Exit `1` is a regression, `2` is a local or invalid-data failure, and `3` is a classified transient primary-provider failure.

**CLI cutoffs and CI thresholds are separate.** The CLI gates precision@1 and optional primary faithfulness; repository CI also gates recall using its locked threshold file. A CLI pass is not the full CI verdict.

**Custom query files are supported.** You can supply your own query set instead of the built-in defaults, which makes it straightforward to benchmark against domain-specific corpora.
