---
type: task
name: benchmark-task
feature: benchmark
depth: task
generated_at: 2026-05-20T03:30:01.573803+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Run the retrieval and faithfulness benchmark

Use the benchmark runner when you need to measure retrieval precision and recall — or optionally faithfulness — and enforce pass/fail thresholds in CI.

## Prerequisites

- Access to the project source code
- A query file if you intend to supply custom queries
- Python dependencies installed so that `pytest` is available on your path

## Run the benchmark

1. **Start a basic retrieval benchmark.**
   Call `main()` in `src/attune_rag/benchmark.py` with no extra flags to evaluate retrieval precision and recall against the default query file:

   ```bash
   python -m attune_rag.benchmark
   ```

2. **Supply a custom query file.**
   Pass your query file path to override the default inputs:

   ```bash
   python -m attune_rag.benchmark --queries path/to/queries.yaml
   ```

3. **Enable faithfulness scoring.**
   Add `--with-faithfulness` to include faithfulness evaluation alongside retrieval metrics:

   ```bash
   python -m attune_rag.benchmark --with-faithfulness
   ```

4. **Configure CI thresholds.**
   Set the threshold flags to the minimum acceptable scores. The runner exits with code `0` when all metrics meet or exceed the thresholds, and a non-zero code otherwise — making it suitable as a CI gate.

5. **Run the related tests.**
   Verify that your configuration changes have not introduced regressions:

   ```bash
   pytest -k "benchmark"
   ```

## Confirm success

The benchmark run succeeded when `main()` returns `0`. In CI, a `0` exit code tells the pipeline that all configured precision, recall, and faithfulness thresholds passed.

## Key files

- `src/attune_rag/benchmark.py` — entry point containing `main()`
