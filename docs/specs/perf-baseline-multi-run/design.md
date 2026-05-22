# Spec: Multi-run perf-baseline methodology — design

> **Status:** **scoped 2026-05-22** — design decisions confirmed in
> [`tasks.md`](tasks.md#scoping-decisions-confirmed-2026-05-22).
> Sections below are the load-bearing design narrative.

## 1. Methodology

### 1.1 What the locked baseline measures

The locked baseline captures **two** noise terms, not one:

- **Intra-run stdev** — within a single workflow invocation, the
  stdev of N trials on the same runner. This is what the current
  baseline measures.
- **Inter-run stdev** — across K separate workflow invocations
  (different runner instances), the stdev of the K per-invocation
  *means*. This is the noise the per-PR delta-check (a single
  invocation) is actually exposed to.

Both are useful: `intra_run_stdev` diagnoses whether a single CI
runner is misbehaving; `inter_run_stdev` is what the gate threshold
is built from.

### 1.2 Aggregation formula

Let invocation `i ∈ [1, K]` produce trials `t_{i,1} … t_{i,N}` for a
given `(benchmark, axis)`.

```
mean_i           = (1/N) Σ_j t_{i,j}              # per-invocation mean
intra_stdev_i    = stdev(t_{i,1} … t_{i,N})       # per-invocation stdev

mean_of_means    = (1/K) Σ_i mean_i               # locked-baseline mean
intra_run_stdev  = (1/K) Σ_i intra_stdev_i        # average intra-run stdev
inter_run_stdev  = stdev(mean_1 … mean_K)         # stdev of the K means

threshold = mean_of_means + sigma * inter_run_stdev
```

`sigma` reverts to 2.0 (its pre-#75 value) once `inter_run_stdev`
is the noise term; the σ=3.0 hack was compensation for measuring
the wrong noise term, not the right one.

### 1.3 Per-PR delta-check stays unchanged

The delta-check still runs one workflow invocation, N=30 trials,
computes the invocation mean, and compares against `threshold`. No
change to its shape — the change is to what `threshold` means.

This matters because the per-PR check is the cost-sensitive job
(one per PR, every PR). Doubling its runner cost would punish every
contributor. Only the lock-baseline job (rare, manual dispatch)
pays the K-fold cost.

## 2. Schema additions

`docs/specs/downstream-validation/perf-thresholds.json` after the
new methodology lands:

```jsonc
{
  // ── existing top-level fields (backward-compat) ───────────────
  "measured_at": "2026-…Z",
  "commit": "<sha>",
  "runs": 100,                   // = invocations * runs_per_invocation
  "sigma": 2.0,                  // back to 2.0 from the σ=3.0 hack
  "include_llm": true,
  "environment": { … },          // K-fold: see §2.1

  // ── new top-level fields ──────────────────────────────────────
  "invocations": 5,              // K
  "runs_per_invocation": 20,     // N
  "methodology_version": 2,      // 1 = legacy intra-run; 2 = multi-run
  "per_invocation": [            // K entries, diagnostics only
    {
      "invocation_id": "…",      // workflow run id of the i-th dispatch
      "commit": "<sha>",         // sanity-check: all K should match
      "environment": { … },      // per-invocation runner fingerprint
      "metrics": {
        "rag_pipeline_run.cpu": { "mean": …, "stdev": …, "n": 20 },
        …
      }
    }
  ],

  // ── per-metric fields ─────────────────────────────────────────
  "metrics": {
    "rag_pipeline_run.cpu": {
      // existing keys (backward-compat aliases):
      "mean":      0.000601,     // = mean_of_means
      "stdev":     0.000048,     // = inter_run_stdev (aliased)
      "threshold": 0.000697,     // = mean + sigma * inter_run_stdev
      "n":         100,          // = invocations * runs_per_invocation

      // new keys:
      "intra_run_stdev": 0.000032,
      "inter_run_stdev": 0.000048,
      "mean_of_means":   0.000601
    }
  }
}
```

### 2.1 The `environment` block

Under methodology v1 there is one environment fingerprint per
baseline. Under v2 there is one per invocation. The top-level
`environment` block stays — it records the fingerprint of the
**aggregation step** (which itself runs on a runner), with a new
key `environment.is_aggregation_runner: true` for clarity. The K
per-invocation runner fingerprints live under
`per_invocation[i].environment`.

This keeps existing single-fingerprint readers working — they see
*some* environment — while making the multi-runner reality available
to anyone who wants to assert "all K invocations ran on the same
SKU" (they won't; that's the point).

### 2.2 `methodology_version`

New top-level integer. v1 = the current single-invocation baseline;
v2 = multi-run. Consumers that want to be strict can refuse to gate
against v1 baselines once v2 is the standard. The format_perf_delta
script gets a one-line check that warns if it sees a v1 baseline
post-Phase-5.

## 3. Implementation approach (orchestration)

### Option A — Workflow self-dispatch + aggregator job

`lock-baseline` workflow_dispatch enqueues K `gh workflow run`
dispatches of a sibling workflow `perf-lock-one` (or a sub-job
inside the same workflow with a separate trigger key). Each
sibling invocation runs once, uploads its per-invocation JSON as
an artifact. A final aggregator step polls for the K artifacts
(or waits on `gh run watch`), pulls them down, runs the
aggregation, commits the locked baseline.

- Pros: each of the K invocations is a fully-independent workflow
  run on a fully-independent runner allocation. This is the most
  faithful possible model of inter-run noise.
- Cons: most complex orchestration. Polling/wait-for-runs logic in
  a workflow is brittle (`gh run watch` works but is slow and
  surfaces no race-free completion guarantee). Adds an inter-
  workflow secret/token dance.

### Option B — Matrix strategy with K parallel jobs

`lock-baseline` job becomes a matrix with K=5 entries. Each matrix
entry runs the measurement once and uploads its per-invocation
JSON as an artifact. A `needs:` aggregator job pulls all K
artifacts and produces the locked baseline.

- Pros: simplest orchestration. Native GitHub Actions, no
  cross-workflow plumbing. One PR at the end, like today.
- Cons: matrix jobs share the same workflow trigger and **may**
  share runner pool affinity — e.g. GitHub may allocate them onto
  the same VHD generation in the same minute. This is partial
  inter-run noise capture; the inter-runner-SKU dimension is
  underexplored relative to option A.

### Option C — External driver (separate workflow or local script)

A higher-level driver (a third workflow, or `make lock-perf-baseline`
on the maintainer's laptop calling `gh workflow run` K times with
spacing) orchestrates K dispatches of the existing `lock-baseline`
workflow, then collects the K resulting baseline PRs (or artifacts).

- Pros: middle complexity. Each invocation is fully independent
  (option A's faithfulness) without the in-workflow polling
  awkwardness; the maintainer's terminal is the orchestrator.
- Cons: not pushbutton from the Actions UI — requires a maintainer
  with a terminal. Couples the lock cadence to maintainer
  availability rather than a workflow_dispatch button.

### Choice: Option B for v1, with an escape to Option C

For the first implementation we pick **Option B (matrix)**. Two
reasons:

1. Even partial inter-runner capture is a step change from zero
   inter-runner capture. The PR #72 vs #74 swing was ±23 %;
   `inter_run_stdev` across a K=5 matrix should still come out
   materially larger than the current `intra_run_stdev`, because
   the dominant inter-run noise sources (CPU SKU mix, page-cache
   state, ambient steal time) are not perfectly correlated across
   matrix entries even when allocated near-simultaneously.
2. Maintenance cost. Option A's polling/aggregator complexity is
   real risk on a methodology change that already has a
   measurement-validity story to defend. We can validate the
   methodology on Option B's noise model and graduate to Option A
   only if M5's measured false-positive rate stays high.

If M5 shows that matrix-mode `inter_run_stdev` is materially smaller
than what we see between two consecutive non-matrix workflow runs
(i.e. the matrix allocation actually clusters runners), tasks.md M2
gains a follow-up M2-bis: re-implement as Option C, driven from a
maintainer-side `scripts/dispatch_perf_lock.sh`. This is an explicit
fallback, not speculative scope.

## 4. Re-lock cadence

Documented in
`docs/specs/downstream-validation/perf-baseline.md` (alongside the
locked record itself) once the first multi-run lock is in place.
Three triggers:

1. **Code-change re-lock** — any merged change touching a hot path
   (`src/attune_rag/retrieval.py`, `corpus/directory.py`,
   `pipeline.py`, `reranker/`). Unchanged from current practice.
2. **Drift re-lock** — the rolling per-invocation mean over the
   last 10 per-PR delta-check runs drifts > 1.5 × `inter_run_stdev`
   from the locked `mean_of_means`. (Threshold chosen to be
   actionable but not flappy; calibrate against the v1 → v2
   transition data once it exists.)
3. **Release-cycle re-lock** — once per minor-version cut, so the
   "perf claim for vX.Y.0" baseline was measured inside the same
   calendar quarter as the release.

## 5. Backward compatibility & migration

- Methodology v1 baselines already on `main` (the current
  `perf-thresholds.json`) keep working. `format_perf_delta.py` reads
  `metrics[<key>].mean / .stdev / .threshold` — those keys mean the
  same thing in both v1 and v2 outputs.
- The σ rollback (3.0 → 2.0) lands **in the same PR** as the first
  v2 lock, not before. Rolling σ back without v2's wider
  `inter_run_stdev` would re-introduce the false-positive rate that
  PR #75 was patching around.
- `methodology_version: 1` is implied (not written) on the existing
  baseline. The first v2 baseline writes `methodology_version: 2`.
  Consumers should treat absent as `1`.
