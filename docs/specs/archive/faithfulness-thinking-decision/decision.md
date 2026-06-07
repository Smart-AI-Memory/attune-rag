# Phase 2 — Faithfulness `--thinking` Default Decision (locked)

```yaml
decision: "off"
date: 2026-05-16
round: v3
labeled_n: 30
rubric_n: 30
control_ids: [gq-008, gq-011]
controls_drift: 0.0
wins_off: 10
wins_on: 4
ties: 16
bootstrap_iters: 10000
bootstrap_seed: 42
bootstrap_point: 6
bootstrap_ci_low: -1
bootstrap_ci_high: 13
ci_excludes_zero: false
phantom_claim_rate: 0.074
phantom_count: 2
total_on_unsupported: 27
phantom_threshold: 0.10
phantom_detector: "content-word overlap < 0.40"
phantom_detector_note: |
  The substring-based detector was rejected (it produced
  100% phantom rates because the judge legitimately
  paraphrases everything it extracts). The content-word
  overlap heuristic is a soft signal — manual review of
  the 27 on-judge unsupported claims suggests the true
  phantom rate is closer to 25-30%, but the detector
  catches only ~7%. The rubric is INSENSITIVE to phantom
  rate at this verdict because wins_off > wins_on, so
  the gap doesn't affect the decision.
margin_stdev: 0.0189
margin_stdev_threshold: 0.10
variance_K: 8
variance_M: 5
variance_queries: [gq-001, gq-002, gq-004, gq-006, gq-012, gq-014, gq-018, gq-019]
off_stdev_pooled: 0.0276
on_stdev_pooled: 0.0290
verdict_label: "off-forever"
ship_at_version: "0.1.18"
rebaseline: false
artifact: artifacts/calibration/thinking-2026-05-16.json
labels: artifacts/calibration/ground-truth-2026-05-16.md
variance: artifacts/calibration/variance-2026-05-16.json
prior_rounds:
  - {round: v1, n: 8,  off_to_on: 1.5, decision: off}
  - {round: v2, n: 17, off_to_on: 1.2, decision: off}
  - {round: v3, n: 30, off_to_on: 2.5, decision: off}
labeler:
  controls_plus_first_two_shifted: Patrick Roebuck
  remaining_27: Claude Opus 4.7 (under Patrick's explicit delegation, same strict-lens protocol)
notes: |
  The 95% bootstrap CI on (wins_off − wins_on) = [−1, +13]
  INCLUDES 0. This means the labeled-sample signal is
  not statistically distinguishable from zero at n=30.
  The point estimate +6 favors off, and the v3 off-to-on
  win ratio (10:4 = 2.5×) is the widest of the three
  rounds (v1 1.5× → v2 1.2× → v3 2.5×), reversing the
  v1→v2 narrowing trend.

  Judge variance is small: margin_stdev = 0.019 sits far
  below the 0.10 escalation threshold; 5 of 8 random-bucket
  queries had σ=0 across 5 runs in BOTH conditions. The
  gq-017-style ±0.25 swings observed in v1/v2 are NOT
  representative of typical judge behavior — they were
  drawn from the high-shift bucket, which is by definition
  the noisiest part of the distribution.

  Phantom-claim pattern persists qualitatively (manual
  read of ON's 27 flagged claims finds 6-7 true phantoms:
  ON paraphrases the answer into stronger or unrelated
  claims, then flags its own paraphrases — see notes on
  gq-010, gq-015, gq-031, gq-005, gq-016, gq-030, gq-017).
  This is a SECONDARY signal for the rubric: it would
  matter only if wins_on > wins_off, which is not the
  case here.

  Methodology caveat (recorded in design.md): the
  labeler shifted mid-round from Patrick to the model.
  The 9:4 (rubric: 10:4) ratio is wide enough that this
  shift is unlikely to change the verdict. If a future
  re-evaluation comes in close to the decision boundary,
  re-label a random subset by hand before locking.

  Phase 4 (downstream validation with attune-gui) and
  any future round can revisit this decision. The
  successor spec should be a new directory
  (faithfulness-thinking-decision-v2/) so this locked
  record is preserved.
```
