# Spec: safe-abstention-defaults — tasks

> **Status:** scoping (2026-06-07). Milestones below are **not**
> executable until the entry gate in
> [`requirements.md`](requirements.md#entry-gates) opens (v1.0.0 cut).
> The scoping decisions block is filled at the `/spec` pass.

## Scoping decisions (locked at `/spec` — TBD)

The five open questions
([`requirements.md` "Open questions"](requirements.md#open-questions-for-scoping)
/ [`design.md` §6](design.md#6-open-questions-for-scoping)) are answered
here before M1 runs. Until then they are open:

1. Bundled-default API surface — _TBD_
2. `### Changed` vs `freeze-override` — _TBD_
3. C3 relative heuristic clears both-corpora legit-recall bar? — _TBD_ (decided by M2)
4. Auto-calibration entry-point location — _TBD_
5. POLICY §7 wording — _TBD_

## Milestones

### M0 — Entry-gate verification
- [ ] Confirm v1.0.0 cut is open for scoping.
- [ ] Confirm the `user-corpus-onboarding` cross-link (R4) is landed or
      lands in the same PR series.

### M1 — Reproducible distribution + calibration script
- [ ] Promote the authoring-time probe to
      `scripts/measure_abstention_distributions.py` (reads
      `queries.yaml` + `queries_negative.yaml` + bundled corpus +
      corpus_b; emits the §2 table). Covered by a unit test.
- [ ] Re-derive the bundled corpus's calibrated `min_score` against the
      **then-current** corpus (don't hardcode this scaffold's snapshot).
- [ ] Output is committed + reproducible (R5).

### M2 — Relative-heuristic measurement (decides Q3)
- [ ] Measure C3 candidates (per-token, gap, combination) on **both**
      corpora's **legit** sets + negatives, including single-token
      queries (risk §6).
- [ ] Verdict: does any C3 variant clear a legit-recall bar on both
      corpora? If yes → C3 ships as the no-eval-data floor; if no → BYO
      without data keeps the conservative default + doc pointer.

### M3 — Bundled default change (the behavior PR)
- [ ] Implement the chosen C1 mechanism (per Q1).
- [ ] Prove R1 (no regression on `queries.yaml`: P@1/R@3 hold) +
      R2 (false-answer rate drops to target) in CI.
- [ ] Preserve the explicit-`min_score` escape hatch (R-non-functional).
- [ ] CHANGELOG `### Changed` (or `freeze-override` per Q2).

### M4 — BYO auto-calibration entry point
- [ ] Implement the auto-calibration path (per Q4), reusing the Phase 5
      `--calibrate-abstention` logic.
- [ ] Cross-link / hand off to `user-corpus-onboarding` for the
      calibrate-first guide.

### M5 — POLICY + docs
- [ ] POLICY §7 wording per Q5 (factual, no over-committed numeric
      floor — risk §3).
- [ ] README / onboarding docs describe the per-corpus default story.

## Done when

- The bundled exemplar is safe-by-default (false-answer rate at the
  measured target, zero legit-recall regression on the gate set).
- BYO corpora have an honest path (auto-calibrate or conservative
  floor) — no unsafe universal constant shipped.
- The default posture has a single source of truth shared with
  `user-corpus-onboarding`.
- All decisions and the calibration are reproducible from committed
  scripts + inputs.

## Provenance

Opened 2026-06-07 from the [`rag-strengthening`](../rag-strengthening/)
program's defaults gap: the abstention mechanism shipped opt-in
(Phase 5), leaving the bundled default at the pre-fix `min_score=2.0`.
Anchored by the distribution measurement in
[`design.md` §2](design.md#2-the-measurement-that-anchors-this-spec).
