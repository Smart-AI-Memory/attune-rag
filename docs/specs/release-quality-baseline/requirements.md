# Spec: Release Quality Baseline (attune-rag)

**Status**: complete (Phase 1 of v1.0 roadmap, shipped via PR [#33](https://github.com/Smart-AI-Memory/attune-rag/pull/33) on 2026-05-16; baseline locked at faithfulness ≥ 0.9686, P@1 ≥ 0.95, R@3 ≥ 1.00)

> Roadmap context: [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md) — Phase 1.
>
> - **Owner:** Patrick
> - **Created:** 2026-05-16
> - **Target release:** 0.1.18 or 0.1.19 (gate is additive — no public-API impact)

---

## Phase 1: Requirements

**Status**: complete

### Problem statement

Today `attune-rag-benchmark` exists and prints P@1, R@3,
and faithfulness numbers, but **no release process checks
those numbers**. A PR that silently regresses faithfulness
from 0.78 to 0.65 ships green. The v1.0 roadmap (Phase 2
"eval story landed", Phase 3 "API freeze") assumes a
trustworthy metric gate — without one, every later phase
becomes unverifiable.

The compounding problem is **judge non-determinism**:
prior calibration runs saw single queries swing by 40 +
percentage points between two runs of the same input. A
naive "fail on any regression" rule produces constant
false positives. A naive "fail on > 5 pp drop" rule misses
real regressions sitting inside the noise floor.

The honest fix is a **noise-floor-aware metric gate**:

- Measure aggregate variance per metric empirically.
- Set the gate at `mean − 2σ` per metric.
- Re-measure when anything affecting the judge changes
  (e.g. `--thinking` defaults in Phase 2).

### Scope

**In scope:**

- New script `scripts/measure_baseline_variance.py` that
  runs the benchmark N times back-to-back on an unchanged
  HEAD and emits per-metric mean and standard deviation.
- Locked baseline doc
  `docs/specs/release-quality-baseline/baseline-N.md`
  recording the measurement run (date, commit, N, raw
  numbers, derived thresholds).
- A CI workflow (extension to `.github/workflows/tests.yml`
  or a new `benchmark.yml`) that runs the benchmark on
  every PR against `main` and fails when any metric drops
  below its locked threshold.
- README section quoting the current baseline + a one-line
  link to the methodology doc.
- A re-measurement procedure documented for use whenever a
  judge-affecting change lands.
- Unit tests for the variance script (deterministic inputs
  → deterministic outputs).

**Out of scope (Non-Goals):**

- Changing the benchmark itself, the judge prompt, the
  query set, or the corpus. This spec gates the existing
  numbers; it does not redefine them.
- Per-query gates. Single-query swings of 40 pp are
  expected per the calibration evidence; this spec only
  gates aggregate metrics.
- Faithfulness with `--thinking` on. The current default
  is opt-in; the gate runs against the current default.
  Phase 2 will revisit if `--thinking` defaults change.
- Cross-corpus comparison. One baseline per corpus; for
  v1.0 that's `attune_help`.
- Live dashboards or trend tracking beyond the per-PR
  gate. (Phase 4 may add this.)

### User stories

1. *As an attune-rag maintainer*, I want a PR that
   regresses P@1 by 4 pp to fail CI, so that quality
   regressions can't ship silently.
2. *As an attune-rag maintainer*, I want the gate to
   tolerate a 1 pp wobble between runs of unchanged code,
   so that CI doesn't cry wolf and erode trust in the
   metric.
3. *As a v1.0 reviewer*, I want the README to show a
   numeric baseline and link to the methodology, so I can
   evaluate whether "stable" claims are backed by
   evidence.
4. *As Phase 2 (eval-story) work*, I want a documented
   re-measurement procedure, so flipping the `--thinking`
   default doesn't invalidate the gate forever.

### Edge cases & open questions

| Question / Edge case | Resolution |
|---|---|
| Benchmark hits a transient LLM API error mid-CI run | Retry once; if still failing, mark the run inconclusive (exit 75) — do not block the PR on infrastructure failures. |
| Faithfulness judge cost on every PR | Phase 1 default: run aggregate metrics on every PR; run the faithfulness judge on PRs that touch `src/attune_rag/{retrieval,reranker,expander,pipeline,prompts,eval}/**` or that include `[full-bench]` in the PR title. Tunable. |
| Baseline drifts upward (we get better) | Re-measure on a quarterly cadence; if the new floor would have failed older releases, document and ratchet up. |
| Baseline drifts downward (we get worse but it's deliberate) | A PR can override the threshold for one merge only via `[baseline-update]` in the title; the merge commit must include a new `baseline-N+1.md` row. |
| N too small to be statistically meaningful | Minimum N = 10; target N = 20. Document the chosen N and the resulting σ in the baseline doc. |
| What counts as "judge-affecting"? | Conservatively: changes to `eval/`, `prompts.py`, the corpus, or any change that flips a flag whose default is read by the benchmark. Documented in the re-measure procedure. |
| Should the gate run on `main` post-merge too? | Yes — catches a regression that slipped through the per-PR run (e.g. base branch drifted). Same workflow, no PR comment. |

### Affected layers

- [x] attune-rag — new script, new CI workflow, README addition, spec docs
- [ ] attune-ai — none
- [ ] attune-gui — none
- [ ] attune-help — none
- [ ] attune-author — none
