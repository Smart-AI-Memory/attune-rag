---
type: quickstart
name: benchmark-quickstart
feature: benchmark
depth: quickstart
generated_at: 2026-06-10T06:07:59.724934+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Quickstart: Run Your First Retrieval Benchmark

From an editable attune-rag repository checkout with the bundled corpus installed (`python -m pip install -e ".[attune-help]"`), run `attune-rag-benchmark` to measure retrieval quality. The default query files live in `tests/golden/` and are not included in the wheel.

```sh
attune-rag-benchmark
```

Expected output: precision@1 and recall@k are reported. A normal run exits `0` when the CLI precision cutoff passes; exit `1` is a measured regression, `2` is a local or invalid-data failure, and `3` is a classified transient primary-provider failure.

## Steps

**1. Install the package**

Use the editable install above for the bundled query files and corpus. With a wheel install, pass your own `--queries` path. The console entry point is registered automatically; keyword needs no retriever extra.

**2. Choose a retriever tier**

Pass `--retriever` to target a specific tier:

```sh
attune-rag-benchmark --retriever hybrid
```

Valid values are `keyword`, `hybrid`, and `transformer`, one per run. Install `[embeddings]` to measure hybrid's embedding leg; it can otherwise fall back to keyword-only. Transformer needs `[transformers]` and exits `2` with an install hint when that dependency is absent.

**3. Add faithfulness scoring (optional)**

To score faithfulness in addition to precision and recall, install `[claude]`, configure `ANTHROPIC_API_KEY`, and add `--with-faithfulness`. Answer generation uses API tokens even if the judge uses a subscription:

```sh
attune-rag-benchmark --retriever hybrid --with-faithfulness
```

**4. Calibrate the abstention threshold (optional)**

Run with `--calibrate-abstention` to recommend an absolute keyword-score threshold. It requires legitimate and negative query sets and does not apply the recommendation:

```sh
attune-rag-benchmark --calibrate-abstention --negatives tests/golden/queries_negative.yaml --json calibration.json
```

The calibration receipt marks retrieval and faithfulness `skipped` with reason `calibration_only`; it is not a completed quality-gate measurement.

**5. Save a normal benchmark receipt**

```sh
attune-rag-benchmark --json benchmark.json
```

Completed retrieval and primary faithfulness are saved before later work. Inspect `outcomes` if the run fails.

## What you just did

You ran `attune-rag-benchmark`, selected a retriever tier, and confirmed the benchmark exits `0` on a passing run. The CLI defaults to precision@1 ≥ `0.70` and, when requested, primary faithfulness ≥ `0.85`. Repository CI separately checks stricter locked thresholds, recall, and the query-file hash. A local CLI pass alone does not prove a CI pass.

## Next:

Supply a custom query file to benchmark against your own data — check `attune-rag-benchmark --help` for the flag that points to your query file path.
