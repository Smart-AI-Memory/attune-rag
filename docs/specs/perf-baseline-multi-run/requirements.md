# Spec: Multi-run perf-baseline methodology — requirements

> **Status:** scaffolding — not executable; activates as Phase 5 work.
> **Entry gates inherited from:** Phase 4 close (W4 complete), 0.2.0
> cut landed, [release-quality-baseline](../release-quality-baseline/)
> intact (the perf gate must not regress retrieval-quality plumbing).

## R1 — Inter-run stdev must be captured into the locked baseline

The locked baseline JSON must contain a measured stdev of the
per-invocation means (the **inter-run** noise), not only the within-
invocation stdev (the **intra-run** noise).

Rationale: the per-PR delta-check is a single workflow invocation
exposed to inter-runner variance. The threshold must include that
variance term, or the gate is fundamentally mis-tuned.

Acceptance: `perf-thresholds.json` for any locked baseline contains
both `intra_run_stdev` and `inter_run_stdev` per `(benchmark, axis)`,
and `threshold` is computed from `inter_run_stdev`.

## R2 — JSON schema additions are backward-compatible

`docs/specs/downstream-validation/perf-thresholds.json` is consumed
by:

- `scripts/format_perf_delta.py` (per-PR delta comment).
- `scripts/check_thresholds.py` — currently a quality-baseline tool,
  but the per-PR perf gate may grow a similar reader in W3.1.
- Anything else (dashboards, downstream consumers) that reads
  `metrics[<key>].mean / .stdev / .threshold`.

The new methodology must not break these readers. Specifically:

- `mean` per metric stays present and continues to mean the central
  estimate the gate compares against. (Under the new methodology it
  is the mean-of-means; existing readers can stay agnostic to
  whether the value came from intra- or inter-run aggregation.)
- `stdev` per metric stays present. To avoid silent semantics drift,
  `stdev` is aliased to `inter_run_stdev` once the new methodology
  is in effect — readers see a larger, more honest number, but
  the **field shape** is unchanged.
- `threshold` per metric stays present and continues to be the
  upper bound the gate compares against.
- New fields are added beside the existing ones, never repurposed:
  `intra_run_stdev`, `inter_run_stdev`, `runs_per_invocation`,
  `invocations`. Top-level `runs` stays for backward-compat as
  `invocations * runs_per_invocation`.

Acceptance: a reader that only uses `mean / stdev / threshold`
keeps working unmodified against a baseline locked with the new
methodology.

## R3 — Reproducible across CI runner SKU drift

GitHub Actions allocates `ubuntu-latest` runners from a pool whose
CPU SKU, kernel age, and co-tenancy mix changes day-to-day. The
locked baseline must average over enough invocations to make the
mean-of-means stable relative to that drift — i.e. re-locking on
the same code on a different day should produce a mean-of-means
within `inter_run_stdev / √K` of the prior lock.

Practical minimum: K ≥ 5 invocations per lock (design.md §3
justifies the floor). Below K=5, the stdev-of-means estimate is too
noisy to be useful as a threshold input.

Acceptance: design.md §3 records the chosen K with an explicit
trade-off note; M5 (post-lock validation) measures whether a
re-lock on the same code produces a mean-of-means inside the
predicted band.

## R4 — Documented re-lock cadence

The current methodology has no formal re-lock schedule — re-locks
happen reactively (PR #75, PR #77) when the gate misbehaves. The
new methodology must specify when to re-lock proactively, because
inter-run noise itself can drift as the runner pool evolves.

The cadence must cover:

- **Code-change triggers**: any merged change to retrieval, corpus
  loading, pipeline assembly, or reranker hot paths re-locks.
  (Unchanged from current practice; called out for completeness.)
- **Drift triggers**: a configurable observation window (proposed:
  10 most-recent per-PR delta-check runs) — if the rolling
  per-invocation mean drifts more than 1.5 × `inter_run_stdev` from
  the locked `mean_of_means`, re-lock.
- **Release-cycle triggers**: re-lock once per minor-version cut
  regardless of drift, so the v1.0.0 perf claim has a baseline
  measured within the same calendar window as the release.

Acceptance: design.md §4 enumerates the three triggers, and
tasks.md M3 includes "document re-lock cadence in
`docs/specs/downstream-validation/perf-baseline.md` once the first
multi-run lock is in place".

## R5 — Cost-bounded

K × N trials per lock cannot blow the workflow's 20-min timeout or
multiply LLM spend beyond a small, predictable factor. See
[risks.md](risks.md) for the budget; here the requirement is the
hard ceiling.

Acceptance: a default lock (K=5, N=20, `include_llm=true`) completes
inside the existing 20-min `lock-baseline` timeout and stays inside
risks.md's stated LLM token budget.
