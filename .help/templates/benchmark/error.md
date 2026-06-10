---
type: error
name: benchmark-error
feature: benchmark
depth: error
generated_at: 2026-06-10T06:07:59.711049+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark errors

## Common error signatures

Errors from the benchmark runner fall into a few recurring categories:

- **Missing extras for a retriever tier.** When you pass `--retriever keyword`, `--retriever hybrid`, or `--retriever transformer` and the corresponding optional dependency is not installed, `main()` exits with code `2` and prints an install hint. This is not a Python exception — it is a deliberate early exit.
- **Bad input files.** Passing a custom query file that does not exist or cannot be parsed produces an error before any scoring begins.
- **Threshold misconfiguration.** Supplying values that conflict with `--calibrate-abstention` expectations or with the configured precision/recall/faithfulness thresholds can cause `main()` to exit non-zero without completing a full benchmark run.
- **Faithfulness scoring failure.** Errors specific to optional faithfulness scoring appear only when you run with `--with-faithfulness` and the underlying scorer encounters a problem.

A successful run of `main()` returns `0`. Any other exit code indicates a failure worth investigating.

## How to diagnose

1. **Check the exit code first.** Exit code `2` with an install hint means a retriever tier's extra package is missing — install the extra named in the hint and re-run. Any other non-zero code points to a runtime or configuration failure.

2. **Read the full error output.** `main()` in `attune_rag.benchmark` prints actionable context alongside failures. The message usually names which flag or input triggered the problem.

3. **Isolate the retriever tier.** If the failure is tier-specific, run `main()` with each `--retriever` value in turn (`keyword`, `hybrid`, `transformer`) to confirm which tier fails and whether the issue is a missing extra or a data problem.

4. **Reproduce without faithfulness scoring.** If you are running `--with-faithfulness`, drop that flag and re-run. A clean run without it tells you whether the failure is in the core retrieval benchmark or in the optional faithfulness scorer.

5. **Validate your query file.** If you are supplying a custom query file, confirm it exists on disk and matches the expected format before passing it to `main()`.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
