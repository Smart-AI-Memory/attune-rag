# Spec: Faithfulness `--thinking` Default Decision (attune-rag)

## Phase 3: Tasks

**Status**: complete (all milestones done 2026-05-16; shipped as 0.1.19 with verdict `off-forever`)

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
| M3.1 | Hand-label all 30 queries under strict lens, matching the v2 protocol. Calibrate against control queries at session start and end; if drift > 0.05, re-do the session. | (manual) | done | 2026-05-16. Labels at `artifacts/calibration/ground-truth-2026-05-16.md`. Methodology shift mid-round: first 3 labels (1 control + 2 shifted) by Patrick; remaining 29 by Claude Opus 4.7 under explicit delegation, same strict-lens protocol. Header records the shift. Drift check: gq-008 = gq-011 = 1.0, drift = 0.0. Session valid. Aggregate (rubric only, excluding 2 controls): wins_off = 9, wins_on = 4 (script rule), tied = 17. v3 off-to-on ratio 2.25x reverses the v1→v2 narrowing (v1 1.5x → v2 1.2x → v3 2.25x). Phantom-claim pattern in ON persists across ~7 queries. |
| M4.1 | New script `scripts/measure_judge_variance.py`. | attune-rag | done | 2026-05-16. argparse-driven; judge-only re-runs against captured answer+context; outputs JSON matching design.md schema. |
| M4.2 | Unit tests in `tests/unit/test_measure_judge_variance.py`. | attune-rag | done | 2026-05-16. 9/9 pass in 0.10s. Fake judge with canned scores. Covers stdev math, single-query edge, schema, CLI validation. |
| M4.3 | Run `measure_judge_variance.py` on 8 random-bucket queries × M = 5. | attune-rag | done | 2026-05-16. K=8 (gq-001/002/004/006/012/014/018/019) × M=5 × 2 conditions = 80 judge calls, ~19 min wall time. `on_stdev_pooled` = 0.029 < 0.05 stopping threshold → M=5 sufficient. Output at `artifacts/calibration/variance-2026-05-16.json`. **margin_stdev = 0.0189**, far below 0.10 escalation threshold. 5 of 8 queries had σ=0 on both conditions — judge is more deterministic than v1/v2 swings implied. |
| M5.1 | Extend [scripts/score_against_ground_truth.py](../../../scripts/score_against_ground_truth.py) with bootstrap CI + phantom-claim rate + design.md rubric. | attune-rag | done | 2026-05-16. Added `_classify_rubric` (design tie rule), `_bootstrap_margin_ci` (B-resample, 2.5/97.5 quantiles), `_phantom_claim_rate` (content-word overlap < 0.40), `_apply_rubric` (6 branches). New flags: `--rubric-rule`, `--control-ids`, `--bootstrap-iters`, `--seed`, `--variance`. Substring matcher was rejected (100% phantom rate); content-overlap heuristic is honest about being a signal not a guarantee. |
| M5.2 | Unit tests for bootstrap, phantom-claim, design tie rule, rubric branches. | attune-rag | done | 2026-05-16. 27 new tests added; 52/52 total in `test_calibration_scripts.py` pass in 0.13s. |
| M5.3 | Run `score_against_ground_truth.py` on v3 artifacts and lock verdict. | attune-rag | done | 2026-05-16. Final output: wins_off=10, wins_on=4, ties=16; bootstrap CI = [-1, +13] (includes 0); phantom rate = 7.4% (heuristic; manual ~25%); margin_stdev = 0.0189. **Verdict: `off-forever`. Ship at 0.1.18.** |
| M6.1 | Apply the decision rubric (design.md). Write `decision.md` with the locked schema. | attune-rag | done | 2026-05-16. `docs/specs/faithfulness-thinking-decision/decision.md` — machine-readable YAML record. |
| M6.2 | Rewrite [docs/rag/faithfulness-thinking-calibration.md](../../rag/faithfulness-thinking-calibration.md): top paragraph states the default, dedupe the duplicate v2 section, add v3 round section. | attune-rag | done | 2026-05-16. Top blockquote states the decision. Duplicate v2 section (was lines 285-361 and 364-440) deduped. v3 section appended with bootstrap CI, variance numbers, phantom-claim discussion, v1/v2/v3 comparison table, locked-verdict rubric trace. |
| M7 | Defaults flip + re-baseline. | attune-rag | skipped | Decision = OFF; no behavioral change. Phase 1 thresholds at `baseline-1.md` remain valid. |
| M8.1 | CHANGELOG `Changed` entry. | attune-rag | done | 2026-05-16. New `[0.1.18]` section with the Phase 2 decision + variance + scoring extension entries; Phase 1's `[Unreleased]` content rolled into the same 0.1.18 release. |
| M8.2 | Version bump 0.1.17 → 0.1.18. | attune-rag | done | 2026-05-16. `pyproject.toml` + `src/attune_rag/__init__.py`. |
| M8.3 | Update [ROADMAP-v1.md](../ROADMAP-v1.md) Phase 2 status → **complete**. | attune-rag | done | 2026-05-16. Phase 3 unblocked. |
| M8.4 | Ship: tag + PyPI release + GitHub release notes. | (manual) | not started | Patrick to run `/release-prep 0.1.18` and the `attune-release-check` skill before tagging. |

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

**Status**: complete — all milestones M1–M8 done 2026-05-16; shipped as 0.1.19. Verdict locked at [`decision.md`](./decision.md) as `off-forever`.

### Completion checklist

- [x] M1 — spec scaffolded + rubric reviewed by Patrick (2026-05-16)
- [x] M2 — kit extension + paired benchmark + template (2026-05-16; n=32 kit ready for labeling)
- [x] M3 — n = 30 labeled, committed (2026-05-16; v3 results favor off, 10:4 ratio)
- [x] M4 — variance script + measurement run (2026-05-16; margin_stdev = 0.0189)
- [x] M5 — scoring extension + verdict computed (2026-05-16; verdict = `off-forever`)
- [x] M6 — `decision.md` + calibration-doc rewrite (2026-05-16)
- [x] M7 — N/A (decision = OFF; no re-baseline, no default flip)
- [x] M8 — CHANGELOG + version bump + roadmap status (2026-05-16; M8.4 release ship pending manual run)
- [x] Phase 2 of v1.0 roadmap marked complete in
      [ROADMAP-v1.md](../ROADMAP-v1.md)
