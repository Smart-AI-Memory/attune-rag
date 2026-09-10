---
type: task
name: benchmark-task
feature: benchmark
depth: task
generated_at: 2026-06-10T06:07:59.700599+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Run the retrieval benchmark

Use `attune-rag-benchmark` when you want to measure retrieval precision, recall, and faithfulness against configurable pass/fail thresholds — for example, as a CI gate before merging a change that touches retrieval logic.

## Prerequisites

- An editable attune-rag checkout for the default `tests/golden/` files, or your own `--queries` file; the wheel does not contain those defaults
- `[attune-help]` installed for the default bundled corpus
- A query file if you plan to supply custom queries (optional)
- The appropriate extras installed for your retrieval tier:
  - `keyword` — no extra required
  - `hybrid` — install the `embeddings` extra
  - `transformer` — install the `transformers` extra

## Run the benchmark

1. **Run the benchmark with your chosen retrieval tier.**
   Pass `--retriever` with one of `keyword`, `hybrid`, or `transformer`:

   ```bash
   attune-rag-benchmark --retriever keyword
   ```

   Keyword needs no retriever extra. Transformer exits `2` with an install hint if `[transformers]` is absent. Hybrid can fall back to keyword-only, so install `[embeddings]` before comparing its scores.

2. **Supply a custom query file (optional).**
   If the default queries do not reflect your workload, pass a custom file:

   ```bash
   attune-rag-benchmark --retriever hybrid --queries my_queries.yaml
   ```

3. **Calibrate the abstention threshold (optional).**
   To recommend an absolute keyword-score threshold, use legitimate and negative query sets. Calibration does not apply the recommendation:

   ```bash
   attune-rag-benchmark --calibrate-abstention --negatives tests/golden/queries_negative.yaml --json calibration.json
   ```

   A successful calibration receipt marks retrieval and faithfulness `skipped` with reason `calibration_only`; it is not a quality-gate pass.

4. **Enable faithfulness scoring (optional).**
   Install `[claude]` and configure `ANTHROPIC_API_KEY`, then add `--with-faithfulness`. Answer generation uses API tokens even if the judge uses a subscription:

   ```bash
   attune-rag-benchmark --retriever transformer --with-faithfulness
   ```

5. **Integrate with CI.**
   Save a normal measurement with `--json benchmark.json`. The CLI gates precision@1 (default `0.70`) and optional primary faithfulness (default `0.85`); it reports recall without a CLI gate. Repository CI checks stricter locked thresholds and the query-file hash with `scripts/check_thresholds.py`.

## Verify success

The benchmark run succeeded when:

- The process exits with code `0`.
- For a normal run, requested measurements are complete and the CLI precision and optional primary faithfulness cutoffs pass. Check `outcomes` in the JSON receipt; completed retrieval and primary faithfulness remain available after a later failure.

Exit `1` means a measured regression. Exit `2` means a local, credential, dependency, schema, or invalid-data failure; read the diagnostic rather than assuming a missing extra. Exit `3` means a classified transient primary-provider failure, not a completed faithfulness measurement.
