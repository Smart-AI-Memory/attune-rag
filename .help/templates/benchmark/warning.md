---
type: warning
name: benchmark-warning
feature: benchmark
depth: warning
generated_at: 2026-06-10T06:07:59.717643+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark cautions

## CI gates can block your pipeline on threshold mismatches

`main()` exits with a non-zero code when results fall below configured thresholds, which means a misconfigured threshold or an untested retriever tier can fail your entire CI pipeline. Before you wire `attune-rag-benchmark` into CI, verify that your threshold values reflect realistic baseline scores for your corpus — overly aggressive thresholds will cause spurious failures on every run.

## Risk areas

### Missing extras cause a silent tier skip with exit code 2

When you pass `--retriever transformer` (or another tier whose package extra is not installed), `main()` exits with code 2 and prints an install hint rather than raising an exception. If your CI script treats only exit code 1 as a failure, a missing-extra condition passes silently. Check your pipeline's exit-code handling to ensure exit code 2 is also treated as a failure.

### Abstention threshold calibration changes recall metrics

`--calibrate-abstention` adjusts the threshold at which the system abstains from answering. Calibrating on a small or unrepresentative query set can move the abstention cutoff in a direction that artificially inflates precision while suppressing recall. Run calibration against a query file that reflects your actual workload, and re-run the full benchmark after calibration to confirm the effect on all reported metrics.

### Faithfulness scoring is opt-in and its absence changes what the benchmark measures

`--with-faithfulness` is not enabled by default. A benchmark run without it reports retrieval quality only — precision and recall — and gives no signal about whether retrieved content actually supports the generated answer. If your use case depends on faithful generation, omit `--with-faithfulness` and you may ship a retrieval configuration that scores well on recall but produces unfaithful responses.

### Custom query files silently determine the meaning of every reported metric

The queries you supply define what "good retrieval" means for that run. A query file that does not cover edge cases in your data — short queries, ambiguous terms, or out-of-domain topics — produces benchmark scores that do not generalise. Treat the query file as a first-class input and version-control it alongside your threshold configuration.

## How to avoid problems

1. **Treat exit code 2 as a failure in CI.** Add an explicit check for exit code 2 alongside exit code 1 so that a missing retriever extra does not silently pass your pipeline.

2. **Pin your query file in version control.** Because every metric is relative to the queries you provide, changing the query file between runs makes scores incomparable. Commit the file and reference it by path in your CI configuration.

3. **Run with `--with-faithfulness` before promoting a retriever to production.** Retrieval metrics alone do not capture whether the system produces faithful answers. Use faithfulness scoring at least once per retriever configuration change, even if you omit it from routine CI runs for speed.

4. **Calibrate abstention on a representative sample, then re-benchmark.** After running `--calibrate-abstention`, immediately re-run the full benchmark to observe the effect on precision and recall together, not just the abstention rate in isolation.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
