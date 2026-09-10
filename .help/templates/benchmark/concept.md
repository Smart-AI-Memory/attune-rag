---
type: concept
name: benchmark-concept
feature: benchmark
depth: concept
generated_at: 2026-06-10T06:07:59.695109+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Benchmark

The `attune_rag.benchmark` module is a retrieval and faithfulness benchmark runner that measures your RAG pipeline's quality and can gate CI on configurable thresholds.

## What benchmark measures

The runner evaluates two dimensions of RAG quality:

- **Retrieval quality** — precision and recall across your retriever tier
- **Faithfulness** — whether generated answers stay grounded in retrieved content (opt-in via `--with-faithfulness`)

These two dimensions are independent: you can ship a pipeline that retrieves well but generates poorly, or vice versa. Running both surfaces which layer needs attention.

## Retriever tiers

The `--retriever` flag selects one tier per run: `keyword`, `hybrid`, or `transformer`. Keyword needs no retriever extra. Hybrid uses `[embeddings]` but can fall back to keyword-only if its embedding leg is unavailable. Transformer requires `[transformers]`; a missing dependency produces exit `2` with an install hint. Install the intended tier before comparing scores.

## Abstention calibration

Passing `--calibrate-abstention` with legitimate and negative query sets recommends an absolute keyword-score threshold. It does not change the retriever configuration. With `--json`, successful calibration records a `calibration` result and marks retrieval and faithfulness `skipped` with reason `calibration_only`, even if `--with-faithfulness` was supplied. Calibrating this threshold is a separate concern from raw precision/recall: a pipeline can score well on retrieval but still answer questions it should abstain from.

## How the pieces fit together

```
custom query file  →  benchmark runner (main)
                            │
                  ┌─────────┴──────────┐
             retriever tier        faithfulness scorer
          (keyword/hybrid/           (optional,
           transformer)            --with-faithfulness)
                  │
            precision / recall
            exit 0 (pass) or non-zero (fail)
```

`main()` is the single entry point — it parses arguments, selects the retriever tier, runs queries, scores results, and returns `0` on success. Exit `1` means a measured regression, `2` means a local or invalid-data failure, and `3` means a classified transient failure during the primary provider pass. With `--json`, completed retrieval and primary faithfulness are saved before later work, so a later failure does not erase completed measurements.

The CLI gates precision@1 at `--min-precision` (default `0.70`) and optional primary faithfulness at `--min-faithfulness` (default `0.85`). It reports recall; the repository CI gates recall separately using its locked threshold file.

## When benchmark matters

Run the benchmark when you:

- Add or swap a retriever tier and want to confirm quality did not regress
- Tune the abstention threshold and need a signal on whether the change helped
- Want a CI gate that fails the build automatically when precision or recall drops below a defined threshold
