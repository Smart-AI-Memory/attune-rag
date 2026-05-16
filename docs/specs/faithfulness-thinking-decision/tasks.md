# Spec: Faithfulness `--thinking` Default Decision (attune-rag)

## Phase 3: Tasks

**Status**: approved

### Implementation order

Sequential with one hard checkpoint: **stop after M1 and
confirm the rubric before spending API tokens on M2.** The
n ≥ 30 paired run costs ~30 min and ~120 LLM calls; the
variance pass adds up to ~160 more. Don't burn that budget
on a rubric the user disagrees with.

| # | Task | Layer | Status | Notes |
|---|------|-------|--------|-------|
| M1.1 | Scaffold this spec directory (requirements.md, design.md, tasks.md). | attune-rag | done | 2026-05-16. |
| M1.2 | Patrick reviews requirements.md acceptance criteria + design.md rubric. Edit before M2. | (review) | done | 2026-05-16. Approved as-written; design.md status flipped to `approved`. |
| M2.1 | Extend [scripts/build_calibration_labeling_kit.py](../../../scripts/build_calibration_labeling_kit.py) with `--n-random N` flag (uniform sample from remaining queries after shift+controls). | attune-rag | done | 2026-05-16. Added `--n-random` + `--seed`; `_select_queries` now returns a `(shifted, controls, random)` tuple. Smoke-tested against `thinking-2026-05-15-v2.json` (5+3+5 = 13 blocks, deterministic under `--seed 42`). |
| M2.2 | Unit tests in `tests/unit/test_calibration_scripts.py` covering `--n-random` (deterministic with `--seed`, disjoint from shift+control buckets, errors on N > remaining queries). | attune-rag | done | 2026-05-16. 20/20 pass (0.13 s). Adds 7 new tests: disjointness, seed reproducibility, n_random=0 skip, missing-rng error, overflow error, CLI smoke for the new bucket, negative-counts rejection. Tests of the legacy `_select_queries` return shape updated to the new tuple signature. |
| M2.3 | Run paired benchmark off+on against current HEAD, n = 40 (full golden), with `--with-faithfulness --compare-thinking --json`. Output: `artifacts/calibration/thinking-2026-05-16.json`. | attune-rag | done | 2026-05-16. Wall time ~28 min. Off mean = 0.979 (matches baseline-1), on mean = 0.957, verdict-shift 38/40 = 95 %. Answer+context embedded per query. Log at `thinking-2026-05-16.log`. |
| M2.4 | Generate labeling kit at **n = 32 total = 15 shift + 15 random + 2 controls** (3 controls intended; methodology ceiling capped at 2 — see design.md sample-plan footnote). The 30 shift+random queries feed the rubric (bootstrap CI numerator/denominator); the 2 controls are session-drift scaffolding (start + end), scored separately, excluded from `wins_off`/`wins_on`/`ties`. Output: `ground-truth-2026-05-16.template.md`. | attune-rag | done | 2026-05-16. `--seed 42 --shift-threshold 0.0` (the default 0.05 was a stale v2 floor; design.md says "largest |off−on|" which means top-N by magnitude regardless of threshold). score_against_ground_truth.py extension (M5.1) must skip control IDs when computing the bootstrap. |
| M3.1 | Hand-label all 30 queries under strict lens, matching the v2 protocol. Calibrate against 3 control queries at session start, mid-session, end; if drift > 0.05, re-do the session. | (manual) | not started | ~3–4 hours focused. Save labels as `ground-truth-2026-05-Q.md` (drop `.template`). Commit. |
| M4.1 | New script `scripts/measure_judge_variance.py` — argparse, re-runs the *judge only* M times per query in each condition, emits per-query mean/stdev. | attune-rag | not started | Pure Python; no network in tests (mock the judge). |
| M4.2 | Unit tests in `tests/unit/test_measure_judge_variance.py` — stub judge, canned scores, verify mean/stdev math and CLI error paths (`--runs < 2`, missing `--query-ids`, query not in artifact). | attune-rag | not started | Same fake-subprocess pattern as `test_measure_baseline_variance.py`. |
| M4.3 | Run `measure_judge_variance.py` on 8 random-bucket queries × M = 5. If `on_stdev_pooled` ≥ 0.05, re-run at M = 10. Output: `artifacts/calibration/variance-2026-05-Q.json`. | attune-rag | not started | Cost: ~80–160 judge-only calls. Cheaper than M2.3 because no generator. |
| M5.1 | Extend [scripts/score_against_ground_truth.py](../../../scripts/score_against_ground_truth.py) with bootstrap CI on `(wins_off − wins_on)` (B = 10 000, 95 % CI) and phantom-claim rate computation. | attune-rag | not started | Phantom check: claim text appears verbatim as a substring of the answer string (case-insensitive). Document the substring matcher's known limits in a one-line comment. |
| M5.2 | Unit tests for bootstrap and phantom-claim functions — deterministic seeded resampling, known phantom and non-phantom inputs, edge cases (all-tied sample, zero unsupported flags). | attune-rag | not started | Bootstrap test asserts CI bounds on a known binomial distribution. |
| M5.3 | Run `score_against_ground_truth.py` on `ground-truth-2026-05-Q.md` + `thinking-2026-05-Q.json` (+ `variance-2026-05-Q.json`). Save the printed verdict and numbers; they go in `decision.md`. | attune-rag | not started | One CLI invocation. The output drives M6. |
| M6.1 | Apply the decision rubric (design.md). Write `decision.md` with the locked schema. | attune-rag | not started | Pure documentation. |
| M6.2 | Rewrite [docs/rag/faithfulness-thinking-calibration.md](../../rag/faithfulness-thinking-calibration.md): top paragraph states the default, dedupe the duplicate v2 section (lines 285–361 vs 364–440), add v3 round section, append decisions log. | attune-rag | not started | `/doc-gen` to normalize formatting, manual review before commit. |
| M7.1 | **Only if decision = ON.** Re-measure thresholds following [docs/specs/release-quality-baseline/re-measure.md](../release-quality-baseline/re-measure.md): run `measure_baseline_variance.py --runs 20` with the new default, commit `baseline-2.md`, update `thresholds.json`. | attune-rag | not started | ~3.6 h benchmark wall-time. Add `[baseline-update]` to the follow-up PR title. |
| M7.2 | **Only if decision = ON.** Flip `DEFAULT_THINKING_ENABLED` / `--thinking` defaults in [src/attune_rag/eval/faithfulness.py](../../../src/attune_rag/eval/faithfulness.py) and [src/attune_rag/benchmark.py](../../../src/attune_rag/benchmark.py). Update tests that assert the old default. | attune-rag | not started | Behavioral change. Drives the 0.2.0 bump. |
| M7.3 | **Only if decision = ON.** Public-API audit pass: confirm Phase 3's `__all__` work (parallel-scoped per Decision 3) doesn't conflict. | attune-rag | not started | Touch base with Phase 3 scoping. |
| M8.1 | CHANGELOG `[Unreleased]` `Changed` entry: decision, evidence summary (n, CI, phantom rate), and (if applicable) re-measurement reference. | attune-rag | not started | One sentence + the numbers. |
| M8.2 | Version bump. **0.1.18** if decision = OFF, **0.2.0** if ON. Update `pyproject.toml`, `src/attune_rag/__init__.py` `__version__`. | attune-rag | not started | `attune-release-check` skill enforces sync. |
| M8.3 | Update [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 2 status → **complete** with the date and a one-line summary of the decision. Phase 3 unblocked. | attune-rag | not started | Cross-reference Phase 1's `complete` row for tone. |
| M8.4 | Ship: tag, PyPI release, GitHub Release notes pasting the calibration doc's new top paragraph. | attune-rag | not started | Standard release flow. |

### Dependencies

```
M1 → M2 → M3 → M4 → M5 → M6 → M7 (conditional) → M8
                                                  ↑
                                     M7 skipped if decision = OFF
```

- M1 produces the rubric; without alignment here the rest
  is wasted.
- M2 produces the artifact and the kit; M3 produces the
  labels; both feed M5.
- M4 produces the variance number; M5's bootstrap needs it
  to apply the "margin_stdev > 0.10 → escalate" rule.
- M5 produces the verdict; M6 ships the prose; M7 ships
  the code if the verdict requires it; M8 ships the
  release.

### Testing strategy

#### `test_build_calibration_labeling_kit.py` (new)

The script currently has no unit tests; M2.2 adds them
because the `--n-random` extension is non-trivial and the
existing shift+controls path is also untested.

- `--n-random N` returns N queries disjoint from the shift
  and control buckets.
- `--seed` makes the random draw reproducible.
- `N` larger than the remaining query pool raises a
  clear error.
- Existing `--n-shifted` / `--n-controls` behavior unchanged
  (regression test).

#### `test_measure_judge_variance.py` (new)

- Happy path: stub judge returns canned scores → script
  computes correct mean and stdev per query and condition.
- `--runs < 2` exits 2 with a clear error (stdev requires
  ≥ 2 samples).
- Query ID not present in the artifact: exits 2.
- Output JSON schema matches the design.md spec exactly
  (`measure_at`, `judge_model`, per-query `off` / `on`
  blocks, `aggregate`).
- Pooled stdev math matches `np.std` on the flat list
  (within float tolerance). Document if we use sample
  stdev (ddof=1) vs. population stdev (ddof=0); design.md
  implies sample stdev — be explicit.

#### `test_score_against_ground_truth.py` (extend)

- Bootstrap CI on a known input: 1000 tied + 100
  off-wins yields a CI strictly above 0.
- Phantom-claim detector: claim is a verbatim substring of
  the answer → not phantom. Claim is a paraphrase not in
  the answer → phantom. Case-insensitive substring match.
- Edge: zero on-judge unsupported flags → phantom rate
  defined as 0.0, not NaN.
- Edge: all-tied labeled sample → CI is degenerate; report
  `(0, 0)`, don't crash.

#### Integration (manual)

- M3: labeling session protocol. Control queries at start,
  mid, end. Drift check.
- M5.3: end-to-end on a small synthetic sample (e.g., 5
  hand-crafted labels) before running on the real n = 30
  labels, to catch script bugs cheaply.

### Dependencies (build / runtime)

No new required dependencies. Bootstrap uses
`statistics.fmean` and `random.choices`; phantom detection
uses string ops. All stdlib.

### Rollback plan

The spec lands code only in M2.1 (kit extension), M4.1
(variance script), M5.1 (scoring extension), and conditionally
M7.2 (default flip). Rollback:

- **Decision turns out wrong post-merge:** open a new spec
  round (`docs/specs/faithfulness-thinking-decision-v2/`)
  with the new evidence. Don't amend the locked
  `decision.md` — append a successor.
- **Default flip broke a downstream:** revert M7.2 and
  M7.1's threshold update; ship a 0.2.1 patch with
  `default OFF` restored and a note in the calibration doc.
  Phase 4 (attune-gui) is the early-warning canary here.
- **Variance script has a bug:** the artifact still exists;
  re-run after the fix. The decision is only locked once
  `decision.md` is committed; everything before that is
  scratch work.

---

## Phase 4: Implementation

**Status**: in progress — M1 cleared 2026-05-16 (spec approved); M2 (kit extension + paired benchmark) is the active milestone.

### Completion checklist

- [x] M1 — spec scaffolded + rubric reviewed by Patrick (2026-05-16)
- [x] M2 — kit extension + paired benchmark + template (2026-05-16; n=32 kit ready for labeling)
- [ ] M3 — n = 30 labeled, committed
- [ ] M4 — variance script + measurement run committed
- [ ] M5 — scoring extension + verdict computed
- [ ] M6 — `decision.md` + calibration-doc rewrite
- [ ] M7 — re-baselined + defaults flipped (conditional)
- [ ] M8 — CHANGELOG + version bump + roadmap status + release
- [ ] Phase 2 of v1.0 roadmap marked complete in
      [ROADMAP-v1.md](../ROADMAP-v1.md)
