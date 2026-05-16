# Spec: Faithfulness `--thinking` Default Decision (attune-rag)

**Status**: approved (Phase 2 of v1.0 roadmap, M1 cleared 2026-05-16)

> Roadmap context: [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md) — Phase 2.
> Depends on: Phase 1 ([docs/specs/release-quality-baseline/](../release-quality-baseline/)) — locked thresholds and the re-measurement procedure are inputs to Phase 2's close-out.
>
> - **Owner:** Patrick
> - **Created:** 2026-05-16
> - **Target release:** 0.1.18 (decision = keep OFF) **or** 0.2.0 (decision = flip ON; behavioral change requires minor bump).

---

## Phase 1: Requirements

**Status**: approved

### Problem statement

`FaithfulnessJudge` ships with `--thinking` opt-in. The v0.1.15
calibration round (8 queries, [docs/rag/faithfulness-thinking-calibration.md](../../rag/faithfulness-thinking-calibration.md))
landed on Option B ("keep opt-in") and the v0.1.16 round (17
queries, same doc) confirmed it. Both rounds reached the same
verdict from different signal:

- **v1 (n = 8):** off-closer 3, on-closer 2, tied 3.
  Off-to-on win ratio = 1.5×.
- **v2 (n = 17):** off-closer 6, on-closer 5, tied 6.
  Off-to-on win ratio = 1.2×.

The conclusion is not wrong, but the margin **narrows** as
sample size grows. The calibration doc still tags the decision
"B confirmed" without a quantified confidence interval, and the
sample shows two phenomena that the current methodology does
not separate:

1. **Judge non-determinism is large.** `gq-017` swung
   Δ = +0.182 (on better) in v1 and Δ = −0.250 (on worse) in
   v2 on the *same* query, *same* prompt. The labeling kit
   biases toward high-shift queries by design — but the
   high-shift bucket is partly capturing run-to-run judge
   noise, not real disagreement.
2. **Phantom-claim flags on judge-on.** The v2 round
   documented at least three judge-on failures where the
   flagged "unsupported" claim phrasing did not literally
   appear in the answer (gq-005, gq-002, plus v1's gq-015 and
   gq-008). This is a systematic pattern, not random noise.

Phase 2's job is to convert "B confirmed at small n with high
per-query variance" into one of three definitive end states:

- **B forever:** keep `--thinking` opt-in. Calibration doc
  is updated to say so without hedge.
- **A:** flip `--thinking` ON by default. Requires evidence
  that on-closer beats off-closer by more than the judge's
  measured noise floor, AND that phantom claims do not
  account for the gap.
- **Auto-thresholded:** route based on a per-query feature
  (e.g. answer length, retrieval entropy, context size). Only
  emerges if A is rejected but on wins on a clearly
  identifiable sub-slice.

Whatever lands, the calibration doc's top paragraph stops
saying "pending" and the CHANGELOG records the decision under
`Changed`.

### Scope

**In scope:**

- An expanded ground-truth labeling round at n ≥ 30 hand-labeled
  queries. Either an extension of the v2 kit (17 → 30) or a
  re-sample from the 40-query golden set. Choice made in design.md.
- A **judge-variance measurement** — same N queries re-run
  M ≥ 5 times per condition (off, on). Reports per-query stdev
  and an aggregate stdev for the (off − on) per-query margin.
- A **bootstrap confidence interval** on the off-vs-on closer-
  count margin from the labeled sample. The decision rubric
  uses this CI, not the raw count.
- Updates to [docs/rag/faithfulness-thinking-calibration.md](../../rag/faithfulness-thinking-calibration.md):
  - Top paragraph states one of: `default OFF`, `default ON`,
    or `auto-thresholded`.
  - The duplicate "Ground-truth validation results — v2"
    section currently appearing twice (lines 285–361 and
    lines 364–440) is deduplicated as part of this round.
  - A new section reports the variance measurement and the
    decision rubric.
- Re-measurement of the Phase 1 quality gate, **only if** the
  decision flips ON. Procedure: [docs/specs/release-quality-baseline/re-measure.md](../release-quality-baseline/re-measure.md).
  Produces `baseline-2.md` and updates `thresholds.json`.
- A CHANGELOG `[Unreleased]` entry under `Changed` recording
  the decision and (if applicable) the re-measurement.
- A version bump: **0.1.18** if decision = OFF, **0.2.0** if
  decision = ON (behavioral default change). Phase 3's
  `__all__` audit is *not* a prerequisite for 0.2.0 here —
  per Decision 3 (soft-parallel) Phase 3 scoping may begin
  during Phase 2 but the eval-story decision ships first.

**Out of scope (Non-Goals):**

- Changing the judge prompt, model, or thinking-budget. This
  spec decides whether to *default* an existing flag; it does
  not redesign the judge.
- Per-query routing implementation. The "auto-thresholded"
  end state is a possible *decision*, but if reached, the
  implementation work is a follow-up spec — not bundled in.
- A token-cost analysis for thinking-mode. The v0.1.15 doc
  notes this gap; addressing it requires `FaithfulnessResult.usage`
  capture. Tracked separately.
- Re-litigating Decision 1 (gate threshold = `mean − 2σ`),
  Decision 2 (attune-gui is the gating downstream), or
  Decision 3 (Phase 2 / Phase 3 sequencing). Locked in
  [ROADMAP-v1.md](../ROADMAP-v1.md).
- A full 40-query labeling pass. Out of scope only if the
  n ≥ 30 sample yields a CI that excludes the decision
  threshold; otherwise the spec escalates to n = 40 before
  ruling.

### User stories

1. *As an attune-rag maintainer*, I want the calibration doc's
   top paragraph to state the `--thinking` default without
   hedging, so v1.0 doesn't ship a TODO in its own
   methodology doc.
2. *As an attune-rag maintainer*, I want a quantified CI on
   the off-vs-on margin, so the next time someone proposes
   flipping the default the answer is "the CI is [a, b], here
   is what would move it" — not "we labeled 17 queries once."
3. *As a v1.0 reviewer*, I want to see judge non-determinism
   measured separately from off-vs-on disagreement, so the
   "judges disagree on 80 % of queries" framing is broken
   into "X % is judge noise, Y % is real."
4. *As Phase 3 (API freeze) work*, I want Phase 2 to ship
   before 0.2.0, so the `--thinking` default flip (if it
   happens) doesn't break the API-freeze claim.
5. *As Phase 4 (downstream validation) work*, I want a
   stable, decided faithfulness gate, so attune-gui can pin
   a version of attune-rag knowing the gate semantics won't
   shift under it.

### Edge cases & open questions

| Question / Edge case | Resolution |
|---|---|
| Labeling round costs are large at n = 30 (M re-runs per query × 2 conditions ≈ 300+ judge calls) | Budget the M re-runs as a single batched job; cache prompts; measure cost before committing the re-run plan. Design.md sets M tentatively at 5; rubric is "stop at M = 5 if per-query σ < 0.05, otherwise climb to M = 10." |
| Labels drift over the labeling session (the labeler gets stricter or laxer mid-batch) | Calibrate against a 3-query gold control at session start, mid-session, and session end. If labels on those drift > 0.05, the session is re-done. |
| The v2 kit's high-shift bias inflates non-determinism estimates | Design.md proposes mixing buckets: 50 % high-shift, 50 % random-from-golden. The variance-measurement re-runs use only the random bucket so the noise floor isn't anchored on the noisiest queries. |
| Decision = auto-threshold but we don't ship the router in 0.2.0 | If the labeled sample suggests a clear sub-slice where on > off, the spec lands a "B for now, router scoped for Phase 4" decision rather than blocking on the router itself. |
| Phase 1's `thresholds.json` becomes stale mid-Phase 2 (someone re-measures for an unrelated reason) | Rebase the labeled artifacts on the new HEAD before computing the rubric; the rubric depends on judge behavior, not on threshold values, so the impact is documentation-only. |
| Decision = ON but the re-measured baseline puts a threshold lower than the current locked one (i.e. on is noisier in aggregate) | Document the regression honestly in `baseline-2.md`. ON ships at 0.2.0 only if the bootstrap CI on the labeled-sample closer-count margin clears the rubric *and* aggregate metrics don't drop below the v0.1.17 numbers by more than 1σ of the v0.1.17 noise floor. |
| What if the labeling kit script doesn't support the chosen sample plan? | Extend [scripts/build_calibration_labeling_kit.py](../../../scripts/build_calibration_labeling_kit.py) under M1. The current `--n-shifted` / `--n-controls` flags only cover the shift-biased layout; design.md proposes adding `--n-random` for the mixed-bucket plan. |
| What if labeling reveals the answer or context capture is incomplete | The v0.1.16 `Known gap` section already flags that the JSON now embeds answer + context (PR #26). Re-run on the most recent benchmark artifact, not an older one. If a labeler still cannot decide, mark `verdict: indeterminate` and exclude from the rubric numerator/denominator. Hard cap at 10 % indeterminate before re-running. |

### Affected layers

- [x] attune-rag — labeling round artifacts, calibration doc
      rewrite, possible re-measurement of `thresholds.json`
      and `baseline-2.md`, possible default flip on
      `FaithfulnessJudge.thinking`, possible `--thinking`
      CLI default flip in `attune_rag.benchmark`.
- [ ] attune-ai — none
- [ ] attune-gui — none (Phase 4 will consume the decision,
      but no API change in Phase 2 itself unless the default
      flips ON; even then, the public surface is unchanged).
- [ ] attune-help — none
- [ ] attune-author — none
