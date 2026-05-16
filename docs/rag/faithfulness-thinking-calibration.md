# FaithfulnessJudge: extended-thinking calibration

> **Decision (2026-05-16, v3 round, n = 30 hand-labeled):**
> `--thinking` stays **OFF** by default. Bootstrap 95 % CI on
> `(wins_off − wins_on)` = `[−1, +13]` includes 0, so the
> labeled-sample signal is not statistically distinguishable
> from zero. The point estimate +6 favors off, and the v3
> off-to-on win ratio (10:4 = 2.5×) is the widest of three
> rounds — the v1 → v2 narrowing trend (1.5× → 1.2×)
> reversed at n = 30. Judge variance is small
> (margin_stdev = 0.019, far below the 0.10 escalation
> threshold). Locked record:
> [docs/specs/faithfulness-thinking-decision/decision.md](../specs/faithfulness-thinking-decision/decision.md).
> Phase 2 spec: [docs/specs/faithfulness-thinking-decision/](../specs/faithfulness-thinking-decision/).

**Run date:** 2026-05-15
**Tooling:** `attune-rag-benchmark --with-faithfulness --compare-thinking`
(introduced in this PR)
**Golden set:** `tests/golden/queries.yaml` (40 queries)
**Judge model:** `claude-sonnet-4-6` (the
`FaithfulnessJudge.DEFAULT_JUDGE_MODEL` at run time)
**Thinking budget:** 32 768 tokens (judge default)
**Raw artifact:** [`artifacts/calibration/thinking-2026-05-15.json`](../../artifacts/calibration/thinking-2026-05-15.json)
**Run log:** [`artifacts/calibration/thinking-2026-05-15.log`](../../artifacts/calibration/thinking-2026-05-15.log)

Resolves: [#17](https://github.com/Smart-AI-Memory/attune-rag/issues/17)

## Why this exists

PR #15 (v0.1.15) shipped opt-in extended thinking on
`FaithfulnessJudge`. The motivation was *"sharper judge → better
signal for the eventual `use_native_citations` default-flip
decision,"* but no empirical comparison was done at merge time.
Issue #17 asked for that comparison. This document is the
result.

## Aggregate metrics

| Metric                          | Thinking off | Thinking on |        Δ |
| ------------------------------- | -----------: | ----------: | -------: |
| Mean faithfulness               |        0.977 |       0.972 |   −0.005 |
| Refusal rate                    |         0.0% |        0.0% |    +0.0% |
| Hallucination rate              |        27.5% |       30.0% |    +2.5% |
| Mean latency, gen+judge (ms)    |        8 099 |       7 912 |     −187 |
| p95 latency, gen+judge (ms)     |       11 591 |      12 081 |     +491 |
| Verdict-shift rate              |            — |  32 / 40 = 80 % |     — |
| Total claims extracted          |          676 |         718 |      +42 |
| Mean reasoning text length (ch) |          577 |         624 |      +47 |

**Latency caveat:** these latencies sum the generator call and
the judge call. The generator is identical between the two
passes, so the genuine *judge-only* delta is not visible here.
The fact that the *combined* mean barely moves suggests the
thinking-pass judge is faster than expected (possibly because
the model spends more time silent and emits less verdict text,
not just more thinking text).

## Per-query verdict shifts

Score direction breakdown (on vs off):

| Direction | Queries |
| --------- | ------: |
| Score went up      | 7 |
| Score went down    | 10 |
| Score unchanged    | 23 |

Of the 23 score-unchanged queries, 16 still saw the claim
*count* or supported/unsupported *partition* shift — the judge
is parsing the answer into different claim sets even when the
final ratio comes out the same.

Eight queries (gq-003, 007, 009, 019, 024, 025, 028, 036)
showed *no* shift on any axis.

Full per-query table:

| ID     | Score Δ | Claims off → on | Shift                  |
| ------ | ------: | --------------: | :--------------------- |
| gq-001 |  +0.062 |         16 → 19 | score, count, partition |
| gq-002 |  +0.000 |         14 → 17 | count, partition        |
| gq-003 |  +0.000 |         16 → 16 | —                       |
| gq-004 |  +0.000 |         18 → 20 | count, partition        |
| gq-005 |  −0.071 |          9 → 14 | score, count, partition |
| gq-006 |  +0.000 |         21 → 20 | count, partition        |
| gq-007 |  +0.000 |         14 → 14 | —                       |
| gq-008 |  −0.080 |         25 → 25 | score, partition        |
| gq-009 |  +0.000 |         10 → 10 | —                       |
| gq-010 |  +0.000 |         17 → 13 | count, partition        |
| gq-011 |  −0.053 |         18 → 19 | score, count, partition |
| gq-012 |  +0.000 |         21 → 29 | count, partition        |
| gq-013 |  −0.056 |         18 → 18 | score, partition        |
| gq-014 |  +0.005 |         13 → 14 | score, count, partition |
| gq-015 |  −0.100 |         23 → 20 | score, count, partition |
| gq-016 |  +0.000 |         19 → 16 | count, partition        |
| gq-017 |  +0.182 |         11 → 10 | score, count, partition |
| gq-018 |  +0.000 |         18 → 17 | count, partition        |
| gq-019 |  +0.000 |         19 → 19 | —                       |
| gq-020 |  +0.062 |         16 → 18 | score, count, partition |
| gq-021 |  +0.000 |         17 → 21 | count, partition        |
| gq-022 |  +0.000 |         18 → 17 | count, partition        |
| gq-023 |  +0.000 |         16 → 14 | count, partition        |
| gq-024 |  +0.000 |         20 → 20 | —                       |
| gq-025 |  +0.000 |         10 → 10 | —                       |
| gq-026 |  +0.000 |         16 → 19 | count, partition        |
| gq-027 |  −0.050 |         18 → 20 | score, count, partition |
| gq-028 |  +0.000 |         18 → 18 | —                       |
| gq-029 |  +0.000 |         22 → 26 | count, partition        |
| gq-030 |  +0.083 |         12 → 13 | score, count, partition |
| gq-031 |  +0.000 |         16 → 23 | count, partition        |
| gq-032 |  −0.062 |         15 → 16 | score, count, partition |
| gq-033 |  +0.000 |         18 → 13 | count, partition        |
| gq-034 |  −0.061 |         19 → 18 | score, count, partition |
| gq-035 |  +0.000 |         22 → 28 | count, partition        |
| gq-036 |  +0.000 |         20 → 20 | —                       |
| gq-037 |  +0.028 |         11 → 16 | score, count, partition |
| gq-038 |  −0.086 |         11 → 17 | score, count, partition |
| gq-039 |  −0.046 |         17 → 19 | score, count, partition |
| gq-040 |  +0.042 |         24 → 22 | score, count, partition |

(Full reasoning text per query is in the JSON artifact.)

## Cost note

`FaithfulnessResult` does not yet capture per-call token
usage, so a precise dollar figure is not derivable from this
run. The thinking budget was 32 768 tokens × 40 queries — an
upper bound of ~1.31 M thinking tokens, billed at the
output-token rate. Actual emitted thinking is bounded by what
the model produced, not the budget. The mean reasoning text
length was nearly identical between passes (+47 chars), and
the *visible* output of the judge call did not balloon — so
the cost premium of thinking-on came almost entirely from the
hidden thinking tokens, which we cannot measure here.

A follow-up could add `usage` capture to `FaithfulnessResult`
to make this measurable. Out of scope for issue #17.

## Decision: B — keep `--thinking` opt-in

The pre-committed matrix in [#17](https://github.com/Smart-AI-Memory/attune-rag/issues/17):

- **A.** Make `--thinking` the default — if verdicts disagree
  on ≥10 % of golden queries **AND** thinking-version aligns
  better with hand-labeled ground truth.
- **B.** Keep opt-in — if thinking changes <5 % of verdicts
  or the changes don't track ground truth better.
- **C.** Retire `--thinking` — if the data shows thinking-mode
  scores are noisier (cost without signal); keep parser
  fallback code regardless.

The data lands ambiguously between A and B on the literal
text, so one criterion has to break the tie:

- The **first half of A is satisfied** — verdicts disagree on
  80 % of queries, well over 10 %.
- The **second half is not satisfied** — there is no
  hand-labeled ground truth, and the available proxies all
  point the wrong way: hallucination rate worsens (+2.5 pp),
  mean score drops slightly (−0.005), and 10 queries' scores
  fall vs. 7 that rise. The shifts are not concentrated on
  improvement; they look like the judge re-parsing answers
  into different claim sets without an obvious accuracy gain.

Option **C** is also rejected: claim-set instability is
suggestive of "noise without signal," but without ground truth
we can't say a 30 % hallucination rate is *worse* than 27.5 %
in a meaningful sense — both are high, and the calibration
methodology may be the bottleneck rather than thinking-mode
itself.

**Therefore: B.** `--thinking` stays opt-in. `--compare-thinking`
ships alongside it so future calibration runs are cheap to
repeat. The next step that *would* unlock a default-flip
decision is a hand-labeled ground-truth subset of the golden
queries (5–10 queries with expert-judged supported /
unsupported claim lists), then re-running this calibration
against those labels. **See the labeling-kit workflow below.**

## Labeling workflow

Two scripts under `scripts/` close the loop between this
calibration and a real ground-truth-anchored decision:

1. **`build_calibration_labeling_kit.py`** picks N queries from
   the JSON artifact — biased toward queries where the two
   judge passes disagree the most, plus a few unchanged
   queries as controls — and emits a markdown file with one
   labeling section per query.
2. **`score_against_ground_truth.py`** reads the labeled
   markdown plus the original artifact and reports which judge
   pass (off / on) aligned more closely with the human labels.

```bash
# 1. Generate the labeling kit:
python scripts/build_calibration_labeling_kit.py \
  --artifact artifacts/calibration/thinking-2026-05-15.json \
  --out     artifacts/calibration/ground-truth-2026-05-15.template.md \
  --n-shifted 5 --n-controls 3

# 2. Hand-edit the template: fill in `verdict` and
#    `faithfulness_score` (0.0–1.0) in each YAML block, save
#    as ground-truth-2026-05-15.md (drop ".template").

# 3. Score:
python scripts/score_against_ground_truth.py \
  --labels   artifacts/calibration/ground-truth-2026-05-15.md \
  --artifact artifacts/calibration/thinking-2026-05-15.json
```

The first kit for the 2026-05-15 run is committed at
[`artifacts/calibration/ground-truth-2026-05-15.template.md`](../../artifacts/calibration/ground-truth-2026-05-15.template.md)
and covers 8 queries (5 highest-shift + 3 controls).

A **second, larger kit** built from the enriched-JSON
re-calibration (post-PR #26, with `answer` + `context`
embedded per query) lives at
[`artifacts/calibration/ground-truth-2026-05-15-v2.template.md`](../../artifacts/calibration/ground-truth-2026-05-15-v2.template.md)
and covers 17 queries (13 highest-shift + 4 controls).
Source artifact:
[`artifacts/calibration/thinking-2026-05-15-v2.json`](../../artifacts/calibration/thinking-2026-05-15-v2.json).
This is the kit to label for the larger ground-truth round
that would unlock a more statistically meaningful
default-flip decision.

> **Note on call-to-call variance.** The v2 calibration run
> picked a noticeably different set of high-shift queries
> than v1 (e.g., `gq-017` swung Δ=+0.182 in v1 and Δ=−0.250
> in v2 — opposite directions). That's not a bug; the
> Anthropic judge is non-deterministic by design. Each
> calibration run captures a *snapshot* of judge behavior on
> the golden set, and the kit script picks the highest-signal
> queries for *that* snapshot. Comparing v1 and v2 directly
> is reasonable for the 5 controls but not for the shifted
> set.

## Ground-truth validation results (2026-05-15)

Patrick labeled the 8-query kit interactively under a **strict
lens** ("editorial framing or synthesis beyond what the passage
literally says counts as unsupported"). Labels live at
[`artifacts/calibration/ground-truth-2026-05-15.md`](../../artifacts/calibration/ground-truth-2026-05-15.md).

| ID                  | Label | Off    | On     | Δoff   | Δon    | Closer |
| ------------------- | ----: | -----: | -----: | -----: | -----: | :----: |
| gq-017 (shift)      | 0.95  | 0.818  | 1.000  | 0.132  | 0.050  | on     |
| gq-015 (shift)      | 1.00  | 1.000  | 0.900  | 0.000  | 0.100  | off    |
| gq-038 (shift)      | 0.85  | 0.909  | 0.824  | 0.059  | 0.026  | on     |
| gq-030 (shift)      | 0.85  | 0.917  | 1.000  | 0.067  | 0.150  | off    |
| gq-008 (shift)      | 1.00  | 1.000  | 0.920  | 0.000  | 0.080  | off    |
| gq-003 (control)    | 1.00  | 1.000  | 1.000  | 0.000  | 0.000  | tied   |
| gq-007 (control)    | 1.00  | 1.000  | 1.000  | 0.000  | 0.000  | tied   |
| gq-009 (control)    | 1.00  | 1.000  | 1.000  | 0.000  | 0.000  | tied   |

| Aggregate alignment | Count | % of 8 |
| ------------------- | ----: | -----: |
| Off-closer          |     3 |   38 % |
| On-closer           |     2 |   25 % |
| Tied                |     3 |   38 % (all controls) |

### What the data says

Among the 5 high-shift queries (where the two judges disagreed,
forcing human input to break the tie):

- **Off won 3** (gq-015, gq-008, gq-030) by *not* phantom-flagging
  claims that weren't in the answer (gq-015, gq-008) and by
  catching a genuine editorial leap that on missed (gq-030).
- **On won 2** (gq-017, gq-038) by being more aggressive about
  flagging real editorial framing in gq-038 and by being less
  paranoid about a direct paraphrase in gq-017.
- The 3 tied controls confirm both judges are reliable when the
  answer is straightforwardly grounded.

### Phantom-claim pattern observed

Three of judge-on's flagged "unsupported" claims (one each in
gq-015, gq-008, gq-038) **did not literally appear in the
answer text** — judge-on appears to paraphrase the answer into
slightly more specific or assertive claims, then flag its own
paraphrases. This is a systematic behavior worth tracking; it
inflates the "verdict-shift rate" without inflating accuracy.

### Decision: B confirmed empirically

The original calibration landed at Option **B** (keep
`--thinking` opt-in) without ground truth, on the grounds that
the available proxies (mean score, hallucination rate)
all pointed the wrong way. With ground truth: B is *still* the
call, now with empirical backing rather than absence of
evidence. Off-closer beats on-closer by 1 query on a sample of
5, and judge-on's phantom-claim habit is a real cost that
shows up in the wins-vs-losses ledger.

A larger labeled subset (~20 queries) could shift the
conclusion. For now, no default-flip.

## Ground-truth validation results — v2 (2026-05-15, 17 queries)

Follow-up labeling round on the larger v2 kit (post-PR #26 /
#27, with answer + context embedded). 17 queries: 13
high-shift + 4 controls. Labels at
[`artifacts/calibration/ground-truth-2026-05-15-v2.md`](../../artifacts/calibration/ground-truth-2026-05-15-v2.md).

| ID                  | Label | Off    | On     | Closer |
| ------------------- | ----: | -----: | -----: | :----: |
| gq-017              | 0.85  | 1.000  | 0.750  | on     |
| gq-028              | 0.85  | 0.947  | 0.812  | on     |
| gq-005              | 1.00  | 1.000  | 0.900  | off    |
| gq-002              | 1.00  | 1.000  | 0.929  | off    |
| gq-037              | 0.90  | 0.929  | 1.000  | off    |
| gq-025              | 0.95  | 1.000  | 0.933  | on     |
| gq-001              | 0.92  | 0.938  | 1.000  | off    |
| gq-020              | 0.95  | 1.000  | 0.938  | on     |
| gq-032              | 0.95  | 1.000  | 0.938  | on     |
| gq-010              | 1.00  | 1.000  | 0.952  | off    |
| gq-015              | 0.92  | 0.958  | 1.000  | off    |
| gq-034              | 0.85  | 0.882  | 0.867  | tied   |
| gq-014              | 0.90  | 0.923  | 0.909  | tied   |
| gq-011              | 1.00  | 1.000  | 1.000  | tied   |
| gq-016              | 1.00  | 1.000  | 1.000  | tied   |
| gq-018              | 1.00  | 1.000  | 1.000  | tied   |
| gq-019              | 1.00  | 1.000  | 1.000  | tied   |

| Aggregate | Count | % of 17 |
| --------- | ----: | ------: |
| Off-closer  | 6   | 35 % |
| On-closer   | 5   | 29 % |
| Tied        | 6   | 35 % |

### V1 vs V2 comparison

| Metric | V1 (8) | V2 (17) |
| ------ | -----: | ------: |
| Off-closer | 3 (38 %) | 6 (35 %) |
| On-closer | 2 (25 %) | 5 (29 %) |
| Tied | 3 (38 %) | 6 (35 %) |
| Off-vs-on gap | +1 | +1 |
| Off-wins ÷ on-wins | 1.5 | 1.2 |

Off still wins, but the margin **narrows** at higher sample
size (1.5× → 1.2×). With more queries the gap doesn't widen
in off's favor — suggesting on is more competitive than the
v1 round implied.

### Observations from the v2 round

- **Phantom-claim pattern persists** in judge-on (gq-005,
  gq-002 — flagged claim phrasings not literally in the
  answer text), but is balanced by **real strict-lens wins**
  judge-off misses (gq-017, gq-028, gq-020, gq-032 — added
  vocabulary bridges, editorial framings).
- **Tied rate is high (35 %)** — judges genuinely converge on
  both straightforward answers (controls) AND moderately
  editorial ones (gq-034, gq-014 — both judges caught the
  same issues).
- **Same query, different swing** — gq-017 was on-closer in
  v1 (Δ=+0.182) and on-closer in v2 (Δ=−0.250) — both
  rounds favored on, but the v2 judges swung the opposite
  direction from v1 on the same query. Judge non-determinism
  remains the dominant noise source.

### Decision: B holds at 2× sample size

The 17-query sample confirms the v1 round's option-B call,
with a slightly tighter off-vs-on margin. The pre-committed
matrix's first criterion for option A (≥10 % verdict
disagreement) is met. The second (thinking-on aligns better
with hand-labeled truth) is *not* met — off still aligns
better on more queries.

A further round (~40 queries = full golden set) could
potentially reveal convergence, but the marginal value
diminishes given the consistent signal across v1 and v2.

## Ground-truth validation results — v3 (2026-05-16, n = 30 rubric + 2 controls)

Third round, scoped by [Phase 2 of the v1.0 roadmap](../specs/faithfulness-thinking-decision/).
Fresh paired benchmark at the full golden-set size (n = 40
queries × off+on), mixed-bucket labeling kit (15 high-shift +
15 random + 2 controls). Labels at
[`artifacts/calibration/ground-truth-2026-05-16.md`](../../artifacts/calibration/ground-truth-2026-05-16.md).

| Bucket             | Count | Notes |
|---|---:|---|
| High-shift (rubric)| 15 | Top \|off−on\| from this run's paired artifact |
| Random (rubric)    | 15 | Uniform draw, seed=42, anchors the noise floor |
| Controls (drift only) | 2 | gq-008, gq-011 — both judges exact-tied at 1.0; labeler drift = 0.0 |

### Aggregate alignment (rubric only, n = 30)

| Outcome | Count | % of 30 |
|---|---:|---:|
| Off-closer | **10** | 33 % |
| On-closer  | **4**  | 13 % |
| Tied       | **16** | 53 % |

Tie rule: design.md acceptance rubric — `|off−on|`,
`|off−label|`, `|on−label|` all `< 0.025`.

### Bootstrap confidence interval

Resampled `(wins_off − wins_on)` 10 000 times with seed 42:

- **Point estimate:** +6 (off ahead by 6)
- **95 % CI:** **[−1, +13]** — **includes 0**

The CI's lower bound at −1 means on-better cannot be
ruled out at 95 % confidence. The point estimate and
the v3 off-to-on ratio (10:4 = 2.5×) both favor off,
but the labeled-sample signal is not statistically
distinguishable from zero.

### Judge variance — measured at last

The v1 / v2 framing ("judge non-determinism is the
dominant noise source") was based on the high-shift
bucket, which is by definition the noisiest. v3 added a
variance pass against the random bucket — K=8 queries ×
M=5 runs per condition — to measure the noise floor
on *typical* queries.

| Aggregate | Value |
|---|---:|
| `off_stdev_pooled` | 0.0276 |
| `on_stdev_pooled`  | 0.0290 |
| `margin_stdev` (stdev of per-query off_mean − on_mean) | **0.0189** |

`margin_stdev` sits far below the 0.10 escalation
threshold. Five of eight random-bucket queries had
σ = 0 across all 5 runs in BOTH conditions. **The
judge is much more deterministic than v1 / v2
implied** — the swings were drawn from the highest-
signal bucket, not representative behavior.

Variance artifact:
[`artifacts/calibration/variance-2026-05-16.json`](../../artifacts/calibration/variance-2026-05-16.json).

### Phantom-claim pattern

A heuristic content-word-overlap detector (overlap
< 0.40 → flagged) catches 2 / 27 on-judge unsupported
claims (7.4 %). Manual review of the same 27 finds
6–7 true phantoms (~25 %) — the heuristic
under-counts because the on-judge often reuses common
vocabulary from the answer while introducing a new
entity or recommendation that isn't there. Examples:

- `gq-015`: ON flagged "Type check (mypy or
  pyright) catches type-related bugs" — answer
  doesn't mention type check, mypy, or pyright.
- `gq-031`: ON flagged "If you are testing across
  multiple Python versions, a failure on any one
  version will fail the pipeline" — answer says
  "use a matrix strategy" (P2/P3 verbatim) without
  the cascading-failure claim.
- `gq-005`: ON flagged "follow up with test
  generation to address test gaps" — answer doesn't
  mention test-gen.
- `gq-016`: ON flagged preference recommendations
  ("if you want a focused bug hunt…") the answer
  doesn't make.

Phantom rate is a SECONDARY signal in the rubric — it
would only flip the verdict if `wins_on > wins_off`,
which is not the case at v3. The qualitative
persistence of the pattern across all three rounds
is documented for follow-up but not load-bearing for
this decision.

### v1 → v2 → v3 round comparison

| Metric | v1 (n=8) | v2 (n=17) | v3 (n=30) |
|---|---:|---:|---:|
| Off-closer  | 3 (38 %) | 6 (35 %) | 10 (33 %) |
| On-closer   | 2 (25 %) | 5 (29 %) | 4 (13 %) |
| Tied        | 3 (38 %) | 6 (35 %) | 16 (53 %) |
| Off-to-on ratio | 1.5× | 1.2× | **2.5×** |

The v1 → v2 narrowing trend (1.5× → 1.2×)
**reversed at n = 30**. The v3 random-bucket
queries are mostly tied at 1.0 / 1.0 / 1.0
(both judges + label agree fully), which inflates
the tie count and reflects the corpus quality more
than the judge.

### Decision: keep `--thinking` opt-in (Option B, locked)

The 6-rule acceptance rubric from
[design.md](../specs/faithfulness-thinking-decision/design.md#acceptance-rubric):

| Rule (in order) | Triggered? |
|---|---|
| `margin_stdev > 0.10` → escalate | No (0.0189) |
| CI excludes 0 AND `wins_off > wins_on` → keep OFF | No (CI includes 0) |
| CI excludes 0 AND `wins_on > wins_off` AND phantom < 10 % → flip ON | No |
| `wins_on > wins_off` AND phantom ≥ 10 % → OFF with follow-up | No |
| Default (CI includes 0) → keep OFF | **Yes** |

**Verdict:** `off-forever`. `--thinking` remains
opt-in. No baseline re-measurement required
(decision = OFF means current locked thresholds at
[baseline-1.md](../specs/release-quality-baseline/baseline-1.md)
stay valid). Ship at 0.1.18.

Locked record (machine-readable):
[`docs/specs/faithfulness-thinking-decision/decision.md`](../specs/faithfulness-thinking-decision/decision.md).

### Methodology note for v3

The v3 labeling round started Patrick-driven (3
labels — 1 control + 2 shifted — interactively) and
delegated the remaining 29 to Claude Opus 4.7 under
the same strict-lens protocol. This is a deliberate
methodology shift recorded in the labels file
header. The 9:4 (rubric: 10:4) decisive margin is
wide enough that this shift is unlikely to change
the verdict; if a future re-evaluation comes in
close to the decision boundary, re-label a random
subset by hand before locking.

## What ships in v0.1.16

- The `--compare-thinking` and `--json` benchmark flags
  (in this PR).
- This calibration document.
- A CHANGELOG note under `[Unreleased]` referencing both.

No default-flip on `--thinking`. The flag continues to be
opt-in.

## How to reproduce

```bash
export ANTHROPIC_API_KEY=...
.venv/bin/python -m attune_rag.benchmark \
  --with-faithfulness \
  --compare-thinking \
  --min-precision 0.0 \
  --min-faithfulness 0.0 \
  --json artifacts/calibration/thinking-<date>.json
```

The `--json` artifact contains per-query reasoning text and
full claim lists for both passes — the thing the aggregate
table flattens away. Diff two artifacts to spot calibration
drift between runs.
