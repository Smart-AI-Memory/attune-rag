---
type: concept
name: benchmark-concept
feature: benchmark
depth: concept
generated_at: 2026-06-10T06:07:59.695109+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark

The `attune_rag.benchmark` module is a retrieval and faithfulness benchmark runner that measures your RAG pipeline's quality and can gate CI on configurable thresholds.

## What benchmark measures

The runner evaluates two dimensions of RAG quality:

- **Retrieval quality** — precision and recall across your retriever tier
- **Faithfulness** — whether generated answers stay grounded in retrieved content (opt-in via `--with-faithfulness`)

These two dimensions are independent: you can ship a pipeline that retrieves well but generates poorly, or vice versa. Running both surfaces which layer needs attention.

## Retriever tiers

The `--retriever` flag accepts three values — `keyword`, `hybrid`, and `transformer` — each representing a different retrieval strategy. When you select a tier whose optional dependency is not installed, the runner exits with code `2` and prints an install hint rather than failing silently.

This design lets a single CI job benchmark whichever tiers are present without breaking on tiers that aren't.

## Abstention calibration

Passing `--calibrate-abstention` tunes the threshold at which the pipeline declines to answer rather than returning a low-confidence result. Calibrating this threshold is a separate concern from raw precision/recall: a pipeline can score well on retrieval but still answer questions it should abstain from.

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

`main()` is the single entry point — it parses arguments, selects the retriever tier, runs queries, scores results, and returns `0` on success. A non-zero exit code lets CI treat a quality regression as a build failure.

## When benchmark matters

Run the benchmark when you:

- Add or swap a retriever tier and want to confirm quality did not regress
- Tune the abstention threshold and need a signal on whether the change helped
- Want a CI gate that fails the build automatically when precision or recall drops below a defined threshold
