---
type: warning
name: benchmark-warning
feature: benchmark
depth: warning
generated_at: 2026-05-20T03:30:01.588414+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark cautions

## What to watch for

The benchmark module runs retrieval and optional faithfulness scoring, then gates CI on configurable thresholds. Misconfigured thresholds or missing query files cause `main()` to return a non-zero exit code, which fails your CI pipeline silently if the calling script does not check the return value.

## Risk areas

**CI threshold misconfiguration blocks or silently skips gating.** `main()` in `src/attune_rag/benchmark.py` returns `0` on success and a non-zero value on failure. If your CI script does not explicitly check the return value, a threshold violation may go undetected. Conversely, thresholds set too aggressively will cause every run to fail, even when retrieval quality is acceptable.

**Faithfulness scoring is opt-in and easy to omit.** Faithfulness evaluation only runs when you pass `--with-faithfulness`. Omitting the flag produces a benchmark result that measures retrieval precision and recall only. If your quality bar requires faithfulness scoring, the absence of the flag means CI can pass on a model that produces unfaithful answers.

**Custom query files silently determine benchmark coverage.** The benchmark uses whatever query file you supply. A query file that is too small, not representative, or accidentally stale will produce misleadingly high scores. Validate your query file before treating benchmark results as a quality signal.

## How to avoid problems

1. **Check `main()`'s return value in CI.** Treat any non-zero return as a hard failure. In shell scripts, use `set -e` or explicitly check `$?` after invoking the benchmark so threshold violations are never swallowed.

2. **Explicitly pass `--with-faithfulness` when faithfulness matters.** Do not rely on a default. Add `--with-faithfulness` to your standard CI invocation and document any intentional omission so reviewers know the benchmark is retrieval-only.

3. **Version-control your query files.** Store query files alongside the benchmark configuration and review changes to them with the same scrutiny as threshold changes. A query file change that improves scores without improving the model is a coverage regression.

4. **Set thresholds deliberately, not optimistically.** Start thresholds at your current baseline and tighten them incrementally. A threshold set above the current model's capability will fail every run; one set too low provides no protection.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
