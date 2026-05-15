# FaithfulnessJudge: extended-thinking calibration

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
against those labels. That's a separate ticket.

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
