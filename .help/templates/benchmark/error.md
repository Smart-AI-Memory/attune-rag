---
type: error
name: benchmark-error
feature: benchmark
depth: error
generated_at: 2026-06-10T06:07:59.711049+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Benchmark errors

## Common error signatures

Errors from the benchmark runner fall into a few recurring categories:

- **Missing dependencies.** Keyword needs no retriever extra. Transformer requires `[transformers]` and exits `2` with an install hint when it is absent. Hybrid uses `[embeddings]` but can fall back to keyword-only. The default bundled corpus separately requires `[attune-help]`.
- **Bad input files.** Passing a custom query file that does not exist or cannot be parsed produces an error before any scoring begins.
- **Invalid limits or measurements.** `--min-precision` and `--min-faithfulness` must be finite numbers between `0` and `1`; `-k` must be positive. Invalid measured rates, nonfinite latency, or incomplete query results produce exit `2`, without marking that stage completed.
- **Faithfulness scoring failure.** Errors specific to optional faithfulness scoring appear only when you run with `--with-faithfulness` and the underlying scorer encounters a problem.

Exit `0` means the requested mode completed successfully; `1` means a measured regression; `2` means a local, credential, schema, or invalid-data failure; `3` means a classified transient failure during the primary provider pass. Exit `3` does not by itself prove a passing retrieval result.

## How to diagnose

1. **Check the exit code first.** Exit code `2` with an install hint means a retriever tier's extra package is missing — install the extra named in the hint and re-run. For `1`, inspect the measured score and cutoff. For other `2` failures, inspect the local diagnostic. For `3`, check the current JSON receipt before considering a retry.

2. **Read stderr and the JSON outcome.** With `--json report.json`, inspect `outcomes.reason` and any `outcomes.detail`. Controlled local validation errors name the field or file; arbitrary provider exception text is redacted. Completed retrieval and primary faithfulness remain available if a later stage fails.

3. **Isolate the retriever tier.** If the failure is tier-specific, run `main()` with each `--retriever` value in turn (`keyword`, `hybrid`, `transformer`) to confirm which tier fails and whether the issue is a missing extra or a data problem.

4. **Reproduce without faithfulness scoring.** If you are running `--with-faithfulness`, drop that flag and re-run. A clean run without it tells you whether the failure is in the core retrieval benchmark or in the optional faithfulness scorer.

5. **Validate your query file.** If you are supplying a custom query file, confirm it is UTF-8 YAML with a nonempty `queries` list. Each row needs unique nonempty string `id` and nonempty string `query` fields. If present, `expected_in_top_3` must be a list of nonempty path strings. Parser diagnostics identify the file without echoing its contents.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
