# Spec: Multi-run perf-baseline methodology (attune-rag)

> **Status:** **scoped 2026-05-22** — executable when Phase 5 opens
> (after Phase 4 W4 close + 0.2.0 cut + 7-day no-hotfix watch).
> See [`tasks.md`](tasks.md). The spec was authored decision-complete;
> the scoping pass confirmed the decisions and added M0 entry-gate
> verification for symmetry with the other Phase 5 specs.
> **Workstream:** Phase 5 (post-freeze methodology fix).
> **Freeze posture:** docs-only scoping. Implementation touches
> `scripts/measure_perf_baseline.py` + `.github/workflows/perf.yml`,
> both gate plumbing rather than public surface — would have
> invalidated the in-flight baseline if landed during the freeze.

## Problem

The Phase 4 perf gate measures the locked baseline by running N
trials inside **one** workflow invocation on **one** GitHub Actions
runner, then captures the within-run mean and stdev. The per-PR
delta-check then runs N trials in its own one-shot invocation and
compares its mean against `mean + sigma * stdev`.

That captures intra-run noise (GC pauses, page faults, statistician's
short-window jitter on a single runner). It does **not** capture
inter-run noise — the variance between two distinct workflow
invocations on different runner instances with different co-tenants,
kernel ages, cache states, and CPU SKUs.

Concretely: two consecutive perf-workflow runs on
perf-relevant-identical code produced a ±23 % swing on
`rag_pipeline_run.cpu`:

| PR | `rag_pipeline_run.cpu` mean (s) | Delta vs locked baseline |
|---|---:|---:|
| [#72](https://github.com/Smart-AI-Memory/attune-rag/pull/72) | 0.000719 | +33.9 % |
| [#74](https://github.com/Smart-AI-Memory/attune-rag/pull/74) | 0.000595 | +10.8 % |

(Locked baseline at the time: `mean = 0.000537`, σ = 0.000034, σ=2.0,
threshold = 0.000605. PR #74 was inside the gate by 1.7 %; PR #72
tripped it by 19 %. Same retrieval-pipeline code in both.)

PRs [#75](https://github.com/Smart-AI-Memory/attune-rag/pull/75) and
[#77](https://github.com/Smart-AI-Memory/attune-rag/pull/77) widened
σ from 2.0 → 3.0 and bumped N (per-PR 10 → 30, baseline 30 → 50).
That made the gate wide enough to ship the freeze, but it patched
the symptom: an artificially inflated σ that absorbs both real
regressions and inter-runner artifacts. The right fix is to measure
inter-run stdev directly and let σ stay near its original 2.0.

## Approach

Change the lock-baseline workflow so it runs the measurement script
K **separate** times (default K=5) — different runner instances,
different invocations — and records the per-invocation means. The
locked baseline then stores:

- `intra_run_stdev` — the average within-invocation stdev (the
  current methodology, retained for diagnostic purposes).
- `inter_run_stdev` — the stdev **of the K per-invocation means**.
  This is the real noise floor the per-PR delta-check has to clear.
- `threshold` — `mean_of_means + sigma * inter_run_stdev`.

The per-PR delta-check is unchanged shape (still one workflow run,
N trials), but it now compares against a threshold built from a
noise model that **includes** the inter-runner component its single
run is exposed to.

## Expected outcomes

1. The σ=3.0 hack rolls back to σ ≈ 2.0 once `inter_run_stdev` is
   the noise term, because `inter_run_stdev` is the noise the gate
   is actually fighting.
2. False-positive rate on the per-PR delta-check drops materially.
   Hard number TBD post-implementation; M5 measures it.
3. The locked baseline becomes defensible for the v1.0.0 perf claim
   — "our gate tolerates measured inter-runner variance, not an
   ad-hoc σ multiplier".
4. JSON schema stays backward-compatible: `mean`, `stdev`, and
   `threshold` keep their meaning for any existing reader; new
   keys are added beside them.

## What this spec is not

- Not a retroactive critique of [#75](https://github.com/Smart-AI-Memory/attune-rag/pull/75)
  or [#77](https://github.com/Smart-AI-Memory/attune-rag/pull/77).
  Both were the right freeze-window fix; this spec is the right
  post-freeze fix.
- Not a change to the per-PR delta-check measurement (the gate
  *evaluation*). Only the **baseline lock** changes.
- Not a methodology change for the retrieval-quality gate
  (`docs/specs/release-quality-baseline/`), which is a separate
  noise regime (deterministic retrieval, judge-based faithfulness)
  and out of scope.

## Layout

- [requirements.md](requirements.md) — invariants the new methodology
  must satisfy (inter-run stdev capture, backward-compat schema,
  runner-SKU robustness, documented re-lock cadence).
- [design.md](design.md) — proposed methodology, schema additions,
  workflow orchestration options + chosen approach.
- [tasks.md](tasks.md) — M1–M5 implementation plan. Scaffolding;
  not executable until Phase 5 opens.
- [risks.md](risks.md) — cost, LLM spend, runner-pool clustering,
  and re-lock churn risks with mitigations.

## Provenance

Promoted from [Phase-5 backlog item M1](../phase-5-backlog/items.md#methodology-inter-run-noise).
Scaffolded with full context fresh while the PR #72/#74 evidence
and the σ=3.0/N=50 reasoning are still in working memory.
