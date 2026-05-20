---
type: troubleshooting
name: benchmark-troubleshooting
feature: benchmark
depth: troubleshooting
generated_at: 2026-05-20T03:30:01.590545+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798847630f
status: generated
---

# Troubleshoot benchmark

## Before you start

The benchmark module runs retrieval and optional faithfulness scoring, then gates CI on configurable thresholds. Confirm the following before digging deeper:

- You can run `python -m attune_rag.benchmark --help` without error.
- Your query file path is correct and the file is non-empty.
- If you passed `--with-faithfulness`, the faithfulness scorer dependency is installed and reachable.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `main()` raises an exception | Read the full traceback — the file and line number identify the exact raise site in `src/attune_rag/benchmark.py` |
| `main()` returns a non-zero exit code | Check whether a precision, recall, or faithfulness threshold was not met; thresholds are configurable |
| Benchmark passes locally but fails in CI | Compare threshold flags and query file paths between your local invocation and the CI command |
| Faithfulness scoring step is skipped silently | Confirm you passed `--with-faithfulness`; without it, faithfulness scoring does not run |
| Benchmark runs but produces unexpected scores | Verify the query file content — malformed or empty queries produce misleadingly low precision/recall |
| Slow benchmark run | Check whether `--with-faithfulness` is enabled; faithfulness scoring adds a call to an external scorer and is the most expensive step |

## Step-by-step diagnosis

1. **Reproduce the failure with a minimal query file.**
   Create a single-entry query file and run:
   ```bash
   python -m attune_rag.benchmark --query-file minimal.jsonl
   ```
   If the failure reproduces, the problem is not specific to your full dataset.

2. **Enable DEBUG logging.**
   Re-run with verbose output to expose intermediate retrieval results and threshold comparisons:
   ```bash
   python -m attune_rag.benchmark --query-file minimal.jsonl --log-level DEBUG
   ```

3. **Check the exit code explicitly.**
   `main()` returns `0` on success. Any other value means a threshold was not met or an error occurred:
   ```bash
   python -m attune_rag.benchmark --query-file your_queries.jsonl; echo "Exit: $?"
   ```

4. **Run the benchmark test suite.**
   Confirm which paths are currently passing:
   ```bash
   pytest -k "benchmark" -v
   ```
   A failing test that exercises your scenario gives you a reproducible fixture to work from.

5. **Isolate faithfulness scoring.**
   If the failure only occurs with `--with-faithfulness`, run once without it to confirm the retrieval path is healthy:
   ```bash
   python -m attune_rag.benchmark --query-file your_queries.jsonl
   ```

## Common fixes

- **Threshold too strict for your dataset.** If CI is failing because scores fall below threshold, adjust the threshold flags to match your dataset's expected performance. Check the CLI help for the exact flag names:
  ```bash
  python -m attune_rag.benchmark --help
  ```

- **Malformed query file.** A query file with missing fields causes silent scoring errors. Validate its structure against the expected schema before re-running.

- **Faithfulness scorer unavailable.** If `--with-faithfulness` causes an import error or connection failure, confirm the scorer dependency is installed:
  ```bash
  pip show <faithfulness-scorer-package>
  ```
  This fix requires a change outside the benchmark module itself — install or configure the dependency in your environment.

- **Environment drift between local and CI.** If benchmark passes locally but fails in CI, compare dependency versions:
  ```bash
  pip freeze | grep -E "attune|<relevant-deps>"
  ```
  Pin versions in your requirements file to keep environments consistent.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
