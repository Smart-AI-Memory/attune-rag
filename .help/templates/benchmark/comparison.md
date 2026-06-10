---
type: comparison
name: benchmark-comparison
feature: benchmark
depth: comparison
generated_at: 2026-06-10T06:07:59.732347+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Comparison: Benchmark vs alternatives

## What benchmark does

`attune_rag.benchmark` is a precision/recall/faithfulness runner installed as the `attune-rag-benchmark` console script. Its entry point is `main()`, which returns `0` on success and `2` when a required retriever extra is missing. Core capabilities:

- Gates CI pipelines on configurable score thresholds
- Benchmarks each retrieval tier via `--retriever {keyword,hybrid,transformer}`
- Exits `2` with an install hint when a tier's optional dependency is absent
- Accepts custom query files
- Calibrates abstention thresholds via `--calibrate-abstention`
- Adds faithfulness scoring via `--with-faithfulness`

## Feature comparison

| Capability | `attune-rag-benchmark` | Ad-hoc / throwaway script | Orchestration layer |
|---|---|---|---|
| Precision & recall scoring | ✅ Built-in | Manual — you write the math | Depends on what it delegates to |
| Faithfulness scoring | ✅ `--with-faithfulness` | Manual | Not guaranteed |
| Multi-tier retriever comparison | ✅ `keyword`, `hybrid`, `transformer` in one run | One tier at a time, by hand | Possible but indirect |
| CI threshold gating | ✅ Non-zero exit on failure (`2` = missing extra, non-zero = threshold breach) | You wire the exit code yourself | Depends on integration |
| Missing-dependency guidance | ✅ Exits `2` with an install hint | Silent failure or unhandled error | Unknown |
| Abstention calibration | ✅ `--calibrate-abstention` | Manual tuning | Not exposed |
| Custom query files | ✅ Supported | Fully custom | Depends on integration |
| Multi-feature orchestration | ❌ Single-feature scope | ✅ Unconstrained | ✅ Purpose-built for this |
| Exploratory one-off analysis | ⚠️ Overhead for single queries | ✅ Low ceremony | ⚠️ May be over-engineered |

## Tradeoffs

**`attune-rag-benchmark` wins when** you need reproducible, threshold-gated quality signals across multiple retriever tiers. The combination of structured exit codes, built-in scoring math, and `--calibrate-abstention` means you get CI-ready output without writing any measurement infrastructure yourself.

**A throwaway script wins when** you are exploring a single query or doing a one-time sanity check. Wiring up the benchmark runner for a single use adds ceremony with no return.

**The orchestration layer wins when** your problem spans multiple features and `benchmark` is only one step in a larger pipeline. Call `main()` from within that layer rather than duplicating its logic.

## Use `attune-rag-benchmark` when…

| Situation | Recommended choice |
|---|---|
| You need CI to fail on quality regression | `attune-rag-benchmark` — exit codes are purpose-built for this |
| You want to compare `keyword` vs `hybrid` vs `transformer` in one run | `attune-rag-benchmark --retriever {keyword,hybrid,transformer}` |
| You need faithfulness scoring alongside precision/recall | `attune-rag-benchmark --with-faithfulness` |
| You are tuning when the system should abstain from answering | `attune-rag-benchmark --calibrate-abstention` |
| You are doing a one-off exploratory query | Throwaway script — skip the benchmark runner |
| Your work spans multiple features | Orchestration layer — call `main()` from there rather than patching internals |
| A retriever tier's dependency is not installed | `attune-rag-benchmark` will exit `2` and tell you exactly what to install |

**Bottom line:** `attune-rag-benchmark` is the right choice any time you need a repeatable, CI-enforceable quality gate. For exploration or multi-feature workflows, the benchmark runner is either too heavyweight or too narrow — use a script or the orchestration layer instead.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`, `retriever-tiers`
