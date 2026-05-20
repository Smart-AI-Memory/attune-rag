---
type: error
name: benchmark-error
feature: benchmark
depth: error
generated_at: 2026-05-20T03:30:01.582825+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark errors

## Common error signatures

These errors occur when the benchmark runner fails to complete a retrieval or faithfulness evaluation. Common failure points include:

- **Threshold gate failures** — the runner exits with a non-zero code when precision, recall, or faithfulness scores fall below configured thresholds. A successful run returns `0`; any other exit code indicates a gate failure.
- **Invalid or missing query file** — passing a custom query file that does not exist or is malformed typically raises an `OSError` or `ValueError` before scoring begins.
- **Faithfulness scoring errors** — errors specific to `--with-faithfulness` runs, such as a missing model or malformed response, appear only when that flag is set.

## Where errors originate

All errors originate in `main()` (`src/attune_rag/benchmark.py`). Because `main()` is the sole entry point, the exit code and any raised exception come directly from this function.

## How to diagnose

1. **Check the exit code first.** `main()` returns `0` on success. A non-zero exit code in CI means a threshold was not met — check your configured precision, recall, or faithfulness thresholds against the reported scores in the output.

2. **Read the full traceback.** If the process raises an exception rather than returning an exit code, the traceback names the exception type and the line in `benchmark.py` where it was raised. An `OSError` points to a file access problem (query file, output path); a `ValueError` points to a configuration or input validation problem.

3. **Isolate faithfulness scoring.** If the failure only occurs with `--with-faithfulness`, re-run without that flag. If the run succeeds, the problem is specific to the faithfulness scoring path rather than retrieval evaluation.

4. **Enable DEBUG logging.** If the exception message alone is not enough, re-run with logging set to `DEBUG`. Log output emitted just before the failure typically identifies the query, threshold value, or file path that caused the error.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
