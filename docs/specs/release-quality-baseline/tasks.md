# Spec: Release Quality Baseline (attune-rag)

## Phase 3: Tasks

**Status**: draft

### Implementation order

Iterative. Each milestone ends with a green test suite
and a usable artifact. M1 is mandatory before any later
milestone — without measured σ, every threshold is a
guess.

| # | Task | Layer | Status | Notes |
|---|------|-------|--------|-------|
| M1.1 | Write `scripts/measure_baseline_variance.py` — argparse, runs the benchmark N times via subprocess, parses stdout, computes per-metric mean / stdev. | attune-rag | done | Pure stdlib. Stdout-parsing approach (not JSON) since the benchmark's `--json` dump requires `--with-faithfulness` and we need the cheap dry-run too. |
| M1.2 | Unit test in `tests/unit/test_measure_baseline_variance.py` with a fake benchmark subprocess (canned stdouts) → deterministic mean / stdev math. | attune-rag | done | 15 tests, 0.09 s, no live API calls. Cmd-aware fake runner so `git_sha()` doesn't consume the stdout queue. |
| M1.3 | Run `measure_baseline_variance.py --runs 20` against current HEAD (commit recorded in baseline-1.md). | attune-rag | done | 2026-05-16. Single run ≈ 11 min × 20 ≈ 3.6 h. Faithfulness mean = 0.979, σ = 0.0052 (much tighter than calibration single-query swings suggested). Retrieval deterministic: P@1 = 0.95, R@3 = 1.00, σ = 0. Threshold = mean − 2σ → faithfulness 0.9686, P@1 0.95, R@3 1.00. |
| M1.4 | Commit `baseline-1.md` and `thresholds.json` under `docs/specs/release-quality-baseline/`. | attune-rag | done | Locked record of the measurement. |
| M2.1 | Write `scripts/check_thresholds.py` — loads a benchmark JSON dump and a thresholds.json, exits 0 / 1 based on whether all metrics meet threshold. | attune-rag | todo | Pure stdlib; no LLM dependency. |
| M2.2 | Unit tests for `check_thresholds.py` — passing case, failing case, missing metric, mismatched `queries_sha256`. | attune-rag | todo | |
| M2.3 | Helper to format a PR comment body (metric, measured, threshold, delta) — pure function, tested. | attune-rag | todo | Used by M3.2. |
| M3.1 | New workflow `.github/workflows/benchmark.yml` — triggers on pull_request + push to main; runs `attune-rag-benchmark --json`, then `check_thresholds.py`. | attune-rag | todo | M3 — Wire the gate into CI. |
| M3.2 | On failure, post / update a PR comment via `gh pr comment`. | attune-rag | todo | Idempotent: same comment edited, not appended. |
| M3.3 | Conditional faithfulness: skip the judge unless the PR touches `src/attune_rag/{retrieval,reranker,expander,pipeline,prompts,eval}/**` OR the PR title contains `[full-bench]`. | attune-rag | todo | Cost control. |
| M3.4 | Inconclusive handling: a transient API error → retry once → still fails → exit 75 → workflow reports neutral, does not block. | attune-rag | todo | Don't punish PRs for infrastructure flakes. |
| M3.5 | Smoke test: deliberately-bad PR (e.g. break a retriever) confirms the gate fails as expected; revert. | attune-rag | todo | Required by Phase 1 gate. |
| M4.1 | README: add a "Quality baseline" section quoting the current numbers + link to `baseline-1.md`. | attune-rag | todo | |
| M4.2 | Re-measurement procedure: short doc at `docs/specs/release-quality-baseline/re-measure.md` explaining when and how. | attune-rag | todo | Phase 2 will be the first user. |
| M4.3 | CHANGELOG entry under the next patch version (`Added` — "release quality gate" and "noise-floor baseline doc"). | attune-rag | todo | |
| M4.4 | Update `ROADMAP-v1.md` Phase 1 status to **in progress** when M1 starts, **complete** when M3.5 passes. | attune-rag | todo | |

### Dependencies

```
M1 → M2 → M3 → M4
```

- M1 produces the measurement; without it M2's
  thresholds.json doesn't exist.
- M2 produces the gate; without it M3's workflow has
  nothing to call.
- M3 wires the gate into CI.
- M4 is documentation + cross-references; can ship in the
  same patch release as M3.

### Testing strategy

All tests live under `tests/unit/` and run on the existing
CI matrix (Python 3.10–3.13, Ubuntu + macOS).

#### `test_measure_baseline_variance.py`

- Happy path: mocked benchmark subprocess emits N canned
  JSON outputs → script computes correct mean and stdev
  per metric.
- N < 10: exit 2 with a clear error.
- One run fails partway: script aborts with a non-zero
  exit and does not write partial output files.
- `--sigma` override is honored in the resulting
  thresholds.

#### `test_check_thresholds.py`

- All metrics pass: exit 0, no output to stderr.
- One metric fails: exit 1, stderr lists the failing
  metric + measured + threshold.
- `queries_sha256` mismatch: exit 2 with a re-measure
  hint.
- Missing metric in the dump: exit 2.

#### `test_pr_comment_body.py`

- Formatter produces stable markdown for a known input
  (golden test).

#### Integration (manual)

- M3.5: confirm the gate fires on a deliberately-bad PR.
- Confirm a normal PR runs in under N minutes (target:
  ≤ 10 min on the existing CI runner).

### Dependencies (build / runtime)

No new required dependencies. The variance script and the
check script are stdlib-only. `gh` is already available
in GitHub Actions.

### Rollback plan

The gate is additive. Rollback strategy:

- **Workflow misbehaves:** disable `benchmark.yml` via
  the Actions UI; no code revert needed.
- **Threshold is wrong:** land a PR that updates
  `thresholds.json` with `[baseline-update]` in the
  title; the change is reviewable and traceable.
- **Script has a bug:** `git revert` the M1 or M2 commit;
  the workflow becomes a no-op (exits 0 with a warning).

---

## Phase 4: Implementation

**Status**: not started

### Completion checklist (placeholders)

- [ ] M1 — variance script + locked baseline-1.md + thresholds.json
- [ ] M2 — threshold-check script + tests
- [ ] M3 — CI workflow wired, PR-comment-on-fail, conditional faithfulness
- [ ] M3.5 — deliberately-bad PR confirms gate fires
- [ ] M4 — README, re-measure doc, CHANGELOG, ROADMAP status update
- [ ] Phase 1 of v1.0 roadmap marked complete in
      [ROADMAP-v1.md](../ROADMAP-v1.md)
