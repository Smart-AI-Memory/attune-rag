---
type: note
name: benchmark-note
feature: benchmark
depth: note
generated_at: 2026-06-10T06:07:59.729746+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Note: benchmark

## Context

`attune_rag.benchmark` is a precision/recall/faithfulness benchmark runner installed as the `attune-rag-benchmark` console script. Its public surface is a single entry point, `main()`, which returns `0` on success.

The module is designed to gate CI pipelines on configurable quality thresholds rather than serve as a library. You invoke it directly from the command line or call `main()` programmatically; no class instantiation is required.

## Design decisions

**Retrieval tiers are opt-in.** The `--retriever` flag accepts `keyword`, `hybrid`, or `transformer`. If the selected tier's optional dependency is not installed, the runner exits with code `2` and prints an install hint rather than raising an unhandled exception. This keeps the base install lightweight.

**Faithfulness scoring is off by default.** Pass `--with-faithfulness` to enable it. Faithfulness scoring typically requires an additional model call, so omitting it speeds up routine retrieval benchmarks in CI.

**Abstention threshold calibration is built in.** The `--calibrate-abstention` flag lets you tune the threshold at which the retriever declines to answer, without writing a separate calibration script.

**Custom query files are supported.** You can supply your own query set instead of the built-in defaults, which makes it straightforward to benchmark against domain-specific corpora.
