---
type: task
name: benchmark-task
feature: benchmark
depth: task
generated_at: 2026-06-10T06:07:59.700599+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Run the retrieval benchmark

Use `attune-rag-benchmark` when you want to measure retrieval precision, recall, and faithfulness against configurable pass/fail thresholds — for example, as a CI gate before merging a change that touches retrieval logic.

## Prerequisites

- `attune_rag` installed in your environment
- A query file if you plan to supply custom queries (optional)
- The appropriate extras installed for your retrieval tier:
  - `keyword` — no extra required
  - `hybrid` — install the `hybrid` extra
  - `transformer` — install the `transformer` extra

## Run the benchmark

1. **Run the benchmark with your chosen retrieval tier.**
   Pass `--retriever` with one of `keyword`, `hybrid`, or `transformer`:

   ```bash
   attune-rag-benchmark --retriever keyword
   ```

   If the extra for that tier is missing, the command exits with code `2` and prints an install hint.

2. **Supply a custom query file (optional).**
   If the default queries do not reflect your workload, pass a custom file:

   ```bash
   attune-rag-benchmark --retriever hybrid --queries my_queries.yaml
   ```

3. **Calibrate the abstention threshold (optional).**
   To tune when the system abstains rather than returns a low-confidence result, add `--calibrate-abstention`:

   ```bash
   attune-rag-benchmark --retriever hybrid --calibrate-abstention
   ```

4. **Enable faithfulness scoring (optional).**
   To include a faithfulness metric alongside precision and recall, add `--with-faithfulness`:

   ```bash
   attune-rag-benchmark --retriever transformer --with-faithfulness
   ```

5. **Integrate with CI.**
   Add the command to your pipeline. The process exits `0` when all metrics meet their thresholds, and a non-zero code when any threshold is missed, so your CI system can gate on the result automatically.

## Verify success

The benchmark run succeeded when:

- The process exits with code `0`.
- The output shows precision, recall, and (if `--with-faithfulness` was passed) faithfulness scores, each at or above your configured thresholds.

If the process exits with code `2`, install the missing extra for your chosen retrieval tier and re-run.
