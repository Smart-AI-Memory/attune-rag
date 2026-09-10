---
type: warning
name: benchmark-warning
feature: benchmark
depth: warning
generated_at: 2026-06-10T06:07:59.717643+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: verified
---

# Benchmark cautions

## CI gates can block your pipeline on threshold mismatches

`main()` exits with a non-zero code when results fall below configured thresholds, which means a misconfigured threshold or an untested retriever tier can fail your entire CI pipeline. Before you wire `attune-rag-benchmark` into CI, verify that your threshold values reflect realistic baseline scores for your corpus — overly aggressive thresholds will cause spurious failures on every run.

## Risk areas

### Local failures and provider outages need distinct handling

Exit `1` is a measured regression; exit `2` is a local, credential, dependency, schema, or invalid-data failure. Exit `3` is reserved for a classified transient primary-provider failure. Do not treat every nonzero result as an outage or retry it blindly. Repository CI validates the current retrieval receipt before retrying an outage once, and retrieval still gates after an unavailable primary provider.

Keyword needs no retriever extra. Transformer requires `[transformers]` and fails with exit `2` when it is absent. Hybrid can fall back to keyword-only without `[embeddings]`; install the intended dependency before interpreting a hybrid comparison.

### Abstention calibration recommends a threshold without applying it

`--calibrate-abstention` recommends an absolute keyword-score threshold from legitimate and negative query sets. It does not mutate the retriever configuration. A successful calibration exits `0`, but its JSON marks retrieval and faithfulness `skipped` with reason `calibration_only`; do not submit that receipt as a completed quality benchmark. Evaluate a recommendation on representative data before applying it.

### Faithfulness scoring is opt-in and its absence changes what the benchmark measures

`--with-faithfulness` is not enabled by default. A benchmark run without it reports retrieval quality only — precision and recall — and gives no signal about whether retrieved content actually supports the generated answer. If your use case depends on faithful generation, omit `--with-faithfulness` and you may ship a retrieval configuration that scores well on recall but produces unfaithful responses.

### Custom query files silently determine the meaning of every reported metric

The queries you supply define what "good retrieval" means for that run. A query file that does not cover edge cases in your data — short queries, ambiguous terms, or out-of-domain topics — produces benchmark scores that do not generalise. Treat the query file as a first-class input and version-control it alongside your threshold configuration.

## How to avoid problems

1. **Keep failures visible in CI.** Block on measured regressions and local failures. Only a classified primary outage with valid, passing retrieval evidence can receive the repository workflow's unavailable-faithfulness treatment. Save `--json` receipts and inspect `outcomes`.

2. **Pin your query file in version control.** Because every metric is relative to the queries you provide, changing the query file between runs makes scores incomparable. Commit the file and reference it by path in your CI configuration.

3. **Run with `--with-faithfulness` before promoting a retriever to production.** Retrieval metrics alone do not capture whether the system produces faithful answers. Use faithfulness scoring at least once per retriever configuration change, even if you omit it from routine CI runs for speed.

4. **Calibrate abstention on a representative sample.** After applying a recommendation to your retriever, measure precision and recall again. Calibration alone changes no configuration.

5. **Distinguish CLI cutoffs from the locked CI baseline.** A local pass at the default `0.70` precision and `0.85` faithfulness cutoffs does not prove the repository's stricter precision, recall, and faithfulness checks pass. Invalid data or a query-file hash mismatch must be fixed or deliberately re-measured, not hidden by lowering thresholds.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
