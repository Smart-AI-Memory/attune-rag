---
type: faq
name: benchmark-faq
feature: benchmark
depth: faq
generated_at: 2026-05-20T03:30:01.592791+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark FAQ

## What does the benchmark feature do?

It runs retrieval and optional faithfulness benchmarks against your pipeline, gates CI on configurable thresholds, and reports precision and recall metrics. You can supply custom query files and enable faithfulness scoring with the `--with-faithfulness` flag.

## When should I use it?

Use benchmark when you want to measure or enforce retrieval quality — for example, before merging a change that touches your retrieval logic, or when tuning threshold values for a CI quality gate.

## What is the entry point?

Call `main()` from `src/attune_rag/benchmark.py`. It accepts an optional `argv` list (defaults to `sys.argv` when `None`) and returns `0` on success.

## How do I enable faithfulness scoring?

Pass `--with-faithfulness` in your argument list. Without it, the runner evaluates only retrieval precision and recall.

## Can I use my own query file?

Yes. The benchmark runner supports custom query files. Check the CLI help (`main(["--help"])`) for the exact flag name and expected format.

## How do I debug a failing benchmark run?

1. Run `pytest -k "benchmark" -v` to confirm the tests themselves pass.
2. If the tests pass but your run still fails, re-run with logging enabled and add a `logger.debug` call at the suspected failure point.
3. Check that your threshold configuration and query file format match what the runner expects.

## Where is the source code?

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
