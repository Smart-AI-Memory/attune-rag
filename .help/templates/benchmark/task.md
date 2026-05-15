---
type: task
name: benchmark-task
feature: benchmark
depth: task
generated_at: 2026-05-15T18:40:20.593069+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Run the benchmark suite

Use the benchmark runner when you want to measure retrieval and faithfulness quality, gate CI on configurable thresholds, or evaluate a custom set of queries.

## Prerequisites

- Access to the project source code
- `pytest` available in your environment

## Run the benchmark

1. **Open the benchmark entry point.**
   Open `src/attune_rag/benchmark.py` and read the `main()` function signature, its parameters, and its docstring. This tells you which flags are available and what exit code `0` signals (a passing run).

2. **Invoke the runner.**
   Call `main()` directly or run it from the command line. Pass `--with-faithfulness` to enable optional faithfulness scoring in addition to the default precision/recall metrics.

3. **Supply a custom query file if needed.**
   Pass your query file as an argument to target a specific dataset instead of the default one. Confirm the file path is correct before running.

4. **Set your threshold values.**
   Configure the minimum acceptable scores for precision, recall, and (if enabled) faithfulness. The runner exits with a non-zero code when any metric falls below its threshold, which causes a CI job to fail.

5. **Run the suite and check results.**
   Execute the runner with `pytest -k "benchmark"` or call `main()` programmatically. Review the output for per-metric scores and the final exit code.

## Verify success

The benchmark run succeeds when:

- `main()` returns `0`
- All reported metrics meet or exceed your configured thresholds
- No assertion errors appear in the `pytest` output

## Key files

- `src/attune_rag/benchmark.py` — entry point containing `main()`
