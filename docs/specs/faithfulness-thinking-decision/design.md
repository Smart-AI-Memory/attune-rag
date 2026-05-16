# Spec: Faithfulness `--thinking` Default Decision (attune-rag)

## Phase 2: Design

**Status**: approved


### Architecture

#### Components touched

```
artifacts/calibration/
  thinking-2026-05-Q.json              # new: fresh paired off+on run, n>=30 base
  ground-truth-2026-05-Q.template.md   # new: kit output, n>=30
  ground-truth-2026-05-Q.md            # new: labeled, committed
  variance-2026-05-Q.json              # new: M re-runs per query, off+on

docs/rag/
  faithfulness-thinking-calibration.md # rewrite top paragraph + new "Variance & decision"
                                       # section + dedupe the duplicate v2 section

docs/specs/faithfulness-thinking-decision/
  requirements.md                       # this spec
  design.md                             # this file
  tasks.md                              # M1..M5 milestones
  decision.md                           # new: locked record of the final call

docs/specs/release-quality-baseline/
  baseline-2.md                         # new ONLY IF default flips ON
  thresholds.json                       # rewritten ONLY IF default flips ON

scripts/
  build_calibration_labeling_kit.py     # extend: add --n-random bucket
  score_against_ground_truth.py         # extend: add bootstrap CI output
  measure_judge_variance.py             # new: M re-runs per query, per-query sigma

src/attune_rag/
  eval/faithfulness.py                  # possibly: change DEFAULT_THINKING_ENABLED
  benchmark.py                          # possibly: flip --thinking default arg
```

Rationale: the spec's deliverable is **a decision and its
artifacts**, not a new feature. Code changes are conditional
on the decision outcome. The variance measurement script is
the only new tool that lands unconditionally — it's
infrastructure the next calibration round (Phase 4 or
beyond) will also need.

#### Data flow

```
M1. Pick the sample plan (extend v2 vs re-sample golden 40)
M2. Run paired benchmark off+on at n>=30 -> thinking-Q.json
M3. Label kit -> ground-truth-Q.md (hand label, strict lens)
M4. Run measure_judge_variance.py on a random subset of N=8
    -> variance-Q.json  (per-query sigma off, sigma on)
M5. score_against_ground_truth.py + bootstrap CI
    -> printed rubric verdict + decision.md
M6. Rewrite calibration doc top + dedupe v2 section
M7. If decision = ON: re-measure baseline (Phase 1's procedure)
M8. CHANGELOG + version bump + ship
```

### Sample plan

Three options. **Recommendation: Option C (mixed-bucket
re-sample)** — see tradeoffs below.

#### Option A: extend v2 kit from 17 to 30

Add 13 queries to the existing v2 labeling round, biased the
same way (high-shift + controls). Re-use `thinking-2026-05-15-v2.json`.

- Pro: zero new benchmark runs, fastest path to labels.
- Pro: preserves v1->v2->v3 continuity for trend tracking.
- Con: v2's high-shift bias means the new 13 queries are
  also high-shift. The labeled sample is *not* representative
  of typical query behavior — it overweights edge cases.
- Con: cannot separate judge non-determinism from
  off-vs-on disagreement (the shift bucket conflates them).
- Con: artifact is from 2026-05-15; judge model may behave
  differently if Anthropic has updated `claude-sonnet-4-6`
  since.

#### Option B: re-pick 30 from the 40-query golden set, fully random

Run a fresh paired benchmark today; label 30 queries picked
uniformly at random from the 40.

- Pro: representative; no shift bias.
- Pro: fresh artifact captures current judge behavior.
- Con: dilutes the labeling signal — most queries will be
  "both judges agree, both correct," giving us ties.
- Con: throws away the v2 labels (17 queries) entirely.

#### Option C: mixed-bucket re-sample (chosen)

Run a fresh paired benchmark; label 33 queries — 30 for the
rubric plus 3 drift-check controls (decided 2026-05-16):

- **15 high-shift queries** (largest |off - on| from the new
  artifact). These force the labeler to break ties; this is
  where the decision signal lives.
- **15 random queries** drawn uniformly from the remaining
  25 golden queries. These anchor the noise floor and let
  the variance script measure judge non-determinism on
  *typical* queries, not edge cases.
- **Up to 3 control queries** (unchanged on score AND claim
  count between off and on). Labeled at session start,
  mid-session, and session end as a labeler-drift detector —
  NOT included in the rubric numerator or denominator. If
  the labeler's scores on the controls drift by > 0.05
  across the session, the session is re-done.

  **Methodology footnote (2026-05-16 run):** the v3 paired
  artifact yielded only 2 queries qualifying as strict
  controls (the on-pass re-parses claim sets even when
  scores tie — 38 / 40 queries showed *some* shift). The
  v3 session uses 2 controls (session start + session end)
  rather than 3. The drift check stays valid; the cadence
  is coarser. Documented here so a future re-run with a
  different shift distribution doesn't silently lose the
  third control.

- Pro: separates "off vs on disagreement" (high-shift bucket)
  from "judge run-to-run variance" (random bucket).
- Pro: fresh artifact; reuses any v2 labels that happen to
  overlap (likely 5-10 queries) as cheap re-validation.
- Pro: 30 is the lower bound; if rubric is inconclusive
  the spec escalates to 40 (full golden set).
- Con: more benchmark cost than A. One full paired run is
  ~120 LLM calls and ~30 min.
- Con: more labeling effort than the v2 round (17 -> 30).

Extension to [scripts/build_calibration_labeling_kit.py](../../../scripts/build_calibration_labeling_kit.py):
add a `--n-random N` flag that picks N queries uniformly
from the remainder after shift+controls are selected.

### Judge-variance measurement

The core methodology gap in v1/v2: we measured judge
*disagreement* (off vs on) but never measured judge
*variance* (same condition, repeated). gq-017 swinging
+0.182 -> -0.250 across runs proves the variance is
non-trivial — but we don't have a number.

#### New script: `scripts/measure_judge_variance.py`

For each of K queries (drawn from the random bucket of the
labeled sample; K = 8 by default), run the judge M times in
each condition (off, on). Emit per-query mean and stdev for
each condition.

```
python scripts/measure_judge_variance.py \
    --artifact artifacts/calibration/thinking-2026-05-Q.json \
    --query-ids gq-001,gq-007,gq-014,gq-019,gq-021,gq-026,gq-031,gq-036 \
    --runs 5 \
    --out artifacts/calibration/variance-2026-05-Q.json
```

Required flags: `--artifact`, `--query-ids`, `--runs`, `--out`.

The script does **not** re-run the generator — it re-runs
only the judge against the captured answer + context from
the artifact. This is cheap (judge-only) and isolates
judge-side variance from generator-side variance.

Output schema:

```json
{
  "judge_model": "claude-sonnet-4-6",
  "runs": 5,
  "queries": {
    "gq-001": {
      "off": {"mean": 0.93, "stdev": 0.04, "raw": [0.92, 0.95, 0.90, 0.94, 0.94]},
      "on":  {"mean": 0.89, "stdev": 0.08, "raw": [0.83, 0.95, 0.85, 0.92, 0.90]}
    }
    // ...
  },
  "aggregate": {
    "off_stdev_pooled": 0.05,
    "on_stdev_pooled":  0.07,
    "margin_stdev":     0.09   // stdev of per-query (off_mean - on_mean)
  }
}
```

#### Stopping rule for M

- Start at M = 5.
- If `on_stdev_pooled` < 0.05, stop — variance is small
  enough that 5 runs bound it tightly.
- Otherwise climb to M = 10 and re-emit.
- Hard cap: M = 10. The variance estimate is *that variance*;
  if it's high, the decision rubric absorbs that, it doesn't
  fix it.

### Acceptance rubric

The labeled sample produces three numbers:

- `wins_off` = count of queries where off-judge was strictly
  closer to the label.
- `wins_on` = count of queries where on-judge was strictly
  closer.
- `ties` = count of queries where both judges were within
  0.025 of each other and of the label (or both exact-tied).

Bootstrap CI on `(wins_off - wins_on)`: resample the labeled
queries with replacement, B = 10 000 times, compute the
margin each time, take the 2.5 % and 97.5 % quantiles.

**Decision rules** (evaluated in order):

| Rule | Outcome |
|---|---|
| If the bootstrap 95 % CI on `(wins_off − wins_on)` excludes 0 and `wins_off > wins_on` | **B forever — keep OFF.** Calibration doc top paragraph: `default OFF`. Ship at 0.1.18. |
| If the bootstrap 95 % CI on `(wins_off − wins_on)` excludes 0, `wins_on > wins_off`, AND phantom-claim flagged-rate (on-judge flags a claim not literally in the answer) is < 10 % of all on-judge unsupported claims | **A — flip ON.** Calibration doc top paragraph: `default ON`. Re-measure thresholds per Phase 1. Ship at 0.2.0. |
| If `wins_on > wins_off` AND phantom-claim rate >= 10 % | **B holds with a follow-up.** On's apparent gain is partly noise from its own paraphrasing habit. Calibration doc says `default OFF; investigate phantom-claim fix before re-evaluating`. Ship at 0.1.18 with a tracked follow-up issue. |
| If the bootstrap 95 % CI on `(wins_off − wins_on)` **includes 0** AND on shows a clear win on an identifiable sub-slice (e.g. answers > N tokens, or queries with retrieval entropy > X) | **Auto-thresholded scoped, B for now.** Calibration doc says `default OFF; routing scoped for Phase 4`. Ship at 0.1.18. |
| If the bootstrap 95 % CI includes 0 AND no clear sub-slice signal | **B forever — keep OFF.** Calibration doc says `default OFF` with a note that on shows no aggregate benefit at n = 30. Ship at 0.1.18. |
| If the per-query `margin_stdev` (from variance script) exceeds 0.10 | **Inconclusive — escalate to n = 40.** The labeled sample's signal is below the noise floor; full golden set required. Spec does not ship until escalation completes. |

Phantom-claim rate is computed by `score_against_ground_truth.py`
(extended): for each on-judge "unsupported" verdict, check
whether the verbatim claim text appears as a substring in
the answer. Flagged rate = phantom claims / total on-judge
unsupported claims.

### Doc rewrite plan

[docs/rag/faithfulness-thinking-calibration.md](../../rag/faithfulness-thinking-calibration.md)
has accreted two duplicate v2 sections (lines 285–361 and
364–440) and a "B confirmed empirically" framing that
predates Phase 2's rubric. Rewrite plan:

1. **Top paragraph** restated as one of the four decisions
   above, no hedge. Date-stamped.
2. **Dedupe** the duplicate "Ground-truth validation results
   — v2" sections; keep one canonical block.
3. **New section** "Variance & decision (v3 round, n ≥ 30)"
   with: sample-plan choice, M re-runs result, bootstrap CI,
   phantom-claim rate, rubric application.
4. **Decisions log** at the bottom — short table tracking
   v1, v2, v3 verdicts so a future reader doesn't need to
   reconstruct the history from git.

Doc-gen workflow: `/doc-gen` on the calibration doc after
the rewrite to normalize formatting; manual review before
commit.

### Locked decision record

`docs/specs/faithfulness-thinking-decision/decision.md` is
the machine-readable companion to the calibration doc's
prose. Schema:

```yaml
decision: "off" | "on" | "auto-thresholded"
date: 2026-05-DD
labeled_n: 30
wins_off: <int>
wins_on: <int>
ties: <int>
bootstrap_ci_low: <float>
bootstrap_ci_high: <float>
phantom_claim_rate: <float>
margin_stdev: <float>
ship_at_version: "0.1.18" | "0.2.0"
rebaseline: true | false
artifact: artifacts/calibration/thinking-2026-05-Q.json
labels: artifacts/calibration/ground-truth-2026-05-Q.md
variance: artifacts/calibration/variance-2026-05-Q.json
notes: |
  free-form prose
```

Phase 4 work and any future re-evaluation reads this file
to know "what was decided, on what evidence, at what
confidence."

### Tradeoffs & alternatives

#### Sample size

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| n = 17 (just use v2) | Already labeled; zero new work | Margin already known to be narrowing; not enough to lock the decision | No |
| n = 30 (mixed bucket, this spec) | Bootstrap CI viable; cost still ~30 min benchmark + a few hours labeling | Labeling fatigue risk; needs control queries to detect drift | **Yes (default)** |
| n = 40 (full golden set) | Maximum signal | 2× labeling effort; only escalate if n = 30 is inconclusive | **Conditional (escalation)** |

#### Variance measurement target

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| Re-run the full benchmark M times | Captures both generator and judge variance | Generator variance is irrelevant to this decision; expensive | No |
| Re-run only the judge M times against captured answer+context | Isolates judge noise (the thing in question); cheap | Requires the v0.1.16 artifact format (answer+context embedded) | **Yes** |
| Estimate variance from existing v1 vs v2 swing on shared queries | Free | n = 17 overlap is too small; swings driven by labeling, not variance | No |

#### Bootstrap vs analytic CI

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| Bootstrap on the labeled-sample margin | Distribution-free; honest about small n | B = 10 000 resamples; some compute | **Yes** |
| Normal-approximation CI | One-liner | Wins-vs-losses is binomial, normal approx is bad at n = 30 | No |
| Bayesian beta-binomial credible interval | Honest about prior | Requires picking a prior; bikeshed risk | No |

#### Phantom-claim handling

| Option | Pros | Cons | Chosen? |
|---|---|---|---|
| Ignore phantom claims | Simpler rubric | v1 and v2 both saw it; ignoring it would justify a flip ON that we'd regret | No |
| Hard veto: any phantom claim => stay OFF | Conservative | Single flagged false-positive shouldn't override aggregate evidence | No |
| Phantom-claim *rate* threshold (10 % of on-judge unsupported flags) | Balances signal vs. nuisance | The 10 % number is judgment, not derived; document the rationale | **Yes** |

### Cost & timing

One paired benchmark run at n = 30: ~30 min, ~120 judge
calls + ~60 generator calls. Variance pass: K = 8 queries ×
M ≤ 10 × 2 conditions = up to 160 judge calls. Labeling:
~3–4 hours of focused human attention on the kit.

Total LLM cost order-of-magnitude: ~300 calls if M = 10, or
~220 if M = 5. Faithfulness judge runs are the dominant
cost. The Phase 1 CI gate also fires on every PR touching
`src/attune_rag/eval/**` — budget for that on every
labeling-kit follow-up PR.

### Security

- Variance script runs locally with the same
  `ANTHROPIC_API_KEY` the existing tools use. No new secret.
- New artifacts (`thinking-Q.json`, `variance-Q.json`,
  `ground-truth-Q.md`) follow the existing path convention
  under `artifacts/calibration/`. Path-traversal validation
  reuses the helper in [src/attune_rag/dashboard/render.py](../../../src/attune_rag/dashboard/render.py).
- The `decision.md` file is plain YAML in markdown; no
  executable content.

### References

- Phase 1 spec: [docs/specs/release-quality-baseline/](../release-quality-baseline/).
- Re-measurement procedure: [docs/specs/release-quality-baseline/re-measure.md](../release-quality-baseline/re-measure.md).
- Existing calibration doc: [docs/rag/faithfulness-thinking-calibration.md](../../rag/faithfulness-thinking-calibration.md).
- v1 / v2 artifacts: [artifacts/calibration/](../../../artifacts/calibration/).
- Labeling kit: [scripts/build_calibration_labeling_kit.py](../../../scripts/build_calibration_labeling_kit.py).
- Scoring: [scripts/score_against_ground_truth.py](../../../scripts/score_against_ground_truth.py).
- Judge: [src/attune_rag/eval/faithfulness.py](../../../src/attune_rag/eval/faithfulness.py).
- Benchmark entry: [src/attune_rag/benchmark.py](../../../src/attune_rag/benchmark.py).
- Decision 3 (soft-parallel sequencing): [docs/specs/ROADMAP-v1.md](../ROADMAP-v1.md).
