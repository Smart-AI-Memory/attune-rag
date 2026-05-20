# Spec: Multi-run perf-baseline methodology — risks

> **Status:** scaffolding — not executable; activates as Phase 5 work.

## R1 — Runner cost on lock

K=5 invocations × N=20 trials per invocation = **100 trials** per
re-lock, vs **50** under the current σ=3.0/N=50 baseline. So the
LLM-free part of the cost roughly doubles.

In wall-time: the current LLM-free baseline (50 trials of all four
benchmarks) completes in well under a minute on `ubuntu-latest`,
and each matrix entry only does 20 of those trials — so each
matrix entry is **faster** than the current single-invocation lock,
even though the aggregate compute is 2x. The aggregator job adds
~30 s for artifact download + math. Net wall-time per re-lock:
comparable to today, well inside the 20-min job timeout.

**Mitigation:** none needed. Cost is not the binding constraint.

## R2 — LLM cost on `include_llm=true` lock

`llm_reranker_rerank` is the only Anthropic-bound benchmark. Under
the current methodology with N=50 + `include_llm=true`, the lock
makes ~50 reranker calls per lock. Under K=5/N=20, each matrix
entry makes 20 calls × 5 matrix entries = **100 reranker calls per
lock** — exactly 2x the current LLM spend.

Per the re-lock cadence in design.md §4, expect:

- One release-cycle lock per minor-version (~1 per quarter).
- Drift locks: bounded by R5 mitigation (skip the LLM benchmark in
  drift locks).
- Code-change locks: per-merge to a hot path. Historically rare.

So the steady-state LLM spend doubles on a small base. At current
attune-rag development cadence the marginal cost is well below the
monthly Anthropic spend on the rest of the project.

**Mitigation:** the drift-trigger re-lock (design.md §4, trigger 2)
runs with `include_llm=false` by default — the LLM-bound benchmark
has its own gigantic intra-run stdev anyway and is the *least*
sensitive to inter-run noise. Reserve `include_llm=true` for the
code-change and release-cycle re-locks.

## R3 — K matrix entries hitting the same noisy runner SKU

GitHub Actions matrix jobs nominally land on independent runners,
but in practice they often co-allocate within the same VHD pool
within the same minute. If all K=5 of a given lock's matrix entries
happen to land on, say, a freshly-rebooted runner generation with
unusually cold page caches, `inter_run_stdev` may **underestimate**
the true inter-runner noise — locking a threshold that's too tight.

**Mitigation 1:** K ≥ 5 is the floor. K=5 has worst-case 1/120
probability of all entries falling in the same noisy tail
quintile under independent allocation; not zero, but acceptable.

**Mitigation 2:** M3 documents the runner fingerprints of all K
invocations alongside the locked baseline, so post-hoc audit can
spot suspicious clustering.

**Mitigation 3:** M5 explicitly measures whether matrix-mode
`inter_run_stdev` is materially smaller than the swing between two
non-matrix workflow runs on the same code. If it is, design.md §3's
Option-C escape activates.

## R4 — Re-lock churn during the v1 → v2 transition

The σ rollback (3.0 → 2.0) happens in the same PR as the first v2
lock (design.md §5). If the v2 `inter_run_stdev` comes out smaller
than expected — because matrix mode clusters runners — the resulting
σ=2.0 threshold could be tighter than today's σ=3.0 threshold,
which would re-introduce false positives during the very PR that's
supposed to fix them.

**Mitigation:** M3's PR description requires a side-by-side check
that **no** metric's v2 threshold is tighter than its v1 threshold
before σ rollback lands. If any are, σ stays at 3.0 for that PR,
and M5's data informs a follow-up to land the rollback once the
inter-run noise floor is empirically nailed down.

## R5 — Aggregator job is a single point of failure

If any one of the K matrix entries fails (runner allocation timeout,
Anthropic API hiccup with `include_llm=true`, transient network),
the aggregator either operates on K-1 inputs (and produces a
slightly noisier baseline) or fails entirely (no baseline locked).

**Mitigation:** the aggregator accepts K-1 inputs down to a floor
of `K_min = 3`. Below 3 invocations the stdev-of-means estimate is
too unreliable to use; the aggregator hard-fails and the maintainer
re-dispatches. Above 3, the aggregator emits the baseline with a
warning recorded in the locked markdown ("Locked with K=4
invocations; one matrix entry failed: <reason>").

## R6 — Methodology version reader skew

A reader on an old `attune-rag` checkout that's gating against a
v2 baseline on `main` will see new fields it ignores plus
backward-compat-aliased existing fields. That works (R2 in
requirements.md). The reverse — a v2-aware reader against a v1
baseline — is the concern: it expects `inter_run_stdev` and finds
only `stdev`.

**Mitigation:** the v2 schema's `methodology_version` field is
read defensively. Absent or `1` → reader treats `stdev` as
`inter_run_stdev` for compatibility, logs a one-line warning
("gating against v1 baseline; thresholds will be widened until v2
re-lock lands"). Only when v2 is fully on `main` does the warning
get removed.
