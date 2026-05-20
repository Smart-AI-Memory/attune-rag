# Spec: Multi-run perf-baseline methodology — tasks

> **Status:** scaffolding — not executable; activates as Phase 5 work.
> Tasks below are deliberately under-specified at the code level —
> they describe the work shape, not the implementation. The
> implementation-PR author re-reads design.md, then fleshes the M1
> changes into concrete code at execution time.

## Entry gates

Inherited from [requirements.md](requirements.md):

- Phase 4 close (W4 complete).
- 0.2.0 minor-version cut has landed.
- The current locked baseline (v1 methodology, σ=3.0, N=50) is
  intact on `main` and producing per-PR comments without crashes.
- This spec's design.md §3 orchestration choice has been re-affirmed
  (or amended) by the implementer at execution time.

## Implementation order

Iterative. M1 lands first as a pure script change with the
workflow still calling it in legacy mode (no behaviour change in
CI). M2 flips the workflow to multi-run. M3 re-locks. M4 audits
downstream consumers. M5 validates after enough per-PR runs have
accumulated.

| # | Task | Layer | Status | Notes |
|---|------|-------|--------|-------|
| M1 | Extend `scripts/measure_perf_baseline.py` with `--invocations K` and `--per-invocation-out` flags. When `K > 1`, the script's caller is expected to invoke it K separate times via the workflow's matrix; the script itself does **not** orchestrate K — it just learns to write a single per-invocation JSON for an aggregator step. Add a new `scripts/aggregate_perf_baseline.py` that takes K per-invocation JSONs and emits the locked `perf-thresholds.json` + `perf-baseline.md` in the v2 schema. Unit-test the aggregation math (mean-of-means, stdev-of-means, intra-run-stdev averaging) with synthetic per-invocation inputs. | attune-rag | scaffold | Pure stdlib. Backward-compat: invoking `measure_perf_baseline.py` with default args (no `--invocations`) keeps emitting the v1 JSON shape. |
| M2 | Update `.github/workflows/perf.yml` `lock-baseline` job to a `strategy.matrix` of K=5 entries (per design.md §3 Option B). Each matrix entry runs `measure_perf_baseline.py --runs 20 --per-invocation-out <id>.json` and uploads the JSON as an artifact. A `needs:` aggregator job downloads the K artifacts, runs `scripts/aggregate_perf_baseline.py`, commits the v2 locked baseline via the existing bot-PR flow. Preserve the existing `--include-llm` toggle (each of the K matrix entries inherits it). Validate locally with `act` or a draft `workflow_dispatch` on a maintainer fork before merging. | attune-rag | scaffold | Watch for the matrix-job runner-pool clustering noted in design.md §3 — log per-invocation runner fingerprint to surface clustering empirically. |
| M3 | Trigger one lock-baseline workflow_dispatch on `main` post-M2-merge with K=5, N=20, `include_llm=true`. Land the resulting bot-PR. Document the comparison vs the current PR #77 baseline in `docs/specs/downstream-validation/perf-baseline.md`: side-by-side table of `mean / intra_run_stdev / inter_run_stdev / threshold` per metric, plus the re-lock cadence text from design.md §4. Then open a follow-up PR rolling σ back from 3.0 → 2.0 (one-line change in `DEFAULT_SIGMA`); confirm in the PR description that no current metric's threshold tightens below the locked v2 `mean + 2.0 * inter_run_stdev`. | attune-rag | scaffold | This is the visible methodology flip. Land outside of any release-prep window to keep the perf-gate noise floor unambiguous for the release-readiness check. |
| M4 | Audit downstream `perf-thresholds.json` readers. Today: `scripts/format_perf_delta.py` (per-PR comment formatter). Probable: `scripts/check_thresholds.py` (currently quality-baseline only; W3.1 may grow a perf variant). Verify each reader either uses only `mean / stdev / threshold` (backward-compat) or has been updated to consume the new fields. Expected outcome: zero code changes — backward-compat aliases hold — but the audit is recorded in the M4 PR description. | attune-rag | scaffold | Probably a no-op PR (audit only). If the audit finds a tightly-coupled reader, the M4 PR also updates it. |
| M5 | Post-implementation validation. After ≥ 4 weeks of per-PR delta-check runs against the v2 baseline, measure: (a) the false-positive rate of the v2 gate vs the projected false-positive rate of the v1 gate with σ=2.0 over the same period (using the per-PR JSONs from the previous 4 weeks of `delta-check` runs against the v1 σ=3.0 gate); (b) whether matrix-mode `inter_run_stdev` is materially smaller than the swing between two consecutive non-matrix workflow runs on identical code. If (b) shows matrix-mode clustering, open the Option-C follow-up per design.md §3. Record findings in a new `docs/specs/perf-baseline-multi-run/m5-validation.md`. | attune-rag | scaffold | M5 is **explicitly** time-gated, not effort-gated. Don't run M5 inside the same calendar week as M3 — the per-PR signal needs time to accumulate. |

## Out of scope

- Changing the per-PR delta-check measurement shape (still N=30
  trials, one invocation). The change is to what `threshold` means,
  not what the gate measures.
- Changing the retrieval-quality gate
  (`docs/specs/release-quality-baseline/`). Different noise regime,
  different gate, out of scope.
- Adding a third noise term (e.g. inter-day stdev over a rolling
  window). If M5 shows the matrix is clustering and Option C is
  still insufficient, that's its own scoping spec, not a stretch on
  this one.
