# Spec: reranker-evaluation — design

> **Status:** **scoped 2026-05-22.** All seven open questions from §9
> are decided in [`tasks.md`](tasks.md#scoping-decisions-locked-2026-05-22).
> Sections below remain as the design narrative — the recommended
> options are the locked decisions; alternatives are kept as
> historical context for future revisions.

## 1. The measurement question

**Does `LLMReranker` lift P@1 / R@3 on the bundled corpus, and at
what token + latency cost — and what default should v1.0.0 ship?**

Decomposed into three measurable axes:

### 1.1 Retrieval-quality axis

Two runs of the existing 40-baseline + 80-paraphrased query set:

| Run | `rerank` | What it measures |
|---|---|---|
| A | off | Keyword-only retrieval (post-alias-sweep floor) |
| B | on  | Current pipeline default (Claude Haiku rerank) |

Per-run output: P@1, R@3, per-difficulty breakdown, per-query
hit-at-rank-1 boolean.

The **delta** is the lift. Today's expectations going in (to be
*tested* by the measurement, not assumed):

- Baseline P@1 / R@3 are both 1.00 under keyword-only. The reranker
  cannot lift either — they're already at ceiling. Run A and Run B
  should both report 100% / 100% on baseline; if they don't, the
  reranker is *moving* a query off rank-1 it had already won, which
  is a regression to investigate before the verdict locks.
- Paraphrased R@3 is 1.00 under keyword-only. Same ceiling
  consideration.
- Paraphrased P@1 is 0.9125 (73/80) under keyword-only. **This is
  the one axis where the reranker can demonstrably lift.** The 7
  misses are the residual measurement focus — does Claude Haiku, given
  the same paraphrased query and an over-fetched candidate set,
  promote the right document to rank 1?

### 1.2 Token-cost axis

Per-query (and aggregate per 80-query run):

- Mean / p50 / p95 input tokens
- Mean / p50 / p95 output tokens
- Total tokens × Haiku list price → dollar cost per 80-query run

The reranker's input is shaped by `candidate_multiplier` (default
3), the system prompt (~330 tokens), and each candidate's summary +
path. Output is a small JSON array of integers. Order-of-magnitude
expectations going in: ~1500-2500 input tokens per query, ~30 output
tokens, ~$0.0008 per query at Haiku list price (to be confirmed by
measurement, not assumed).

[`perf-baseline.md`](../downstream-validation/perf-baseline.md)
already records reranker latency (`llm_reranker_rerank.wall` mean
0.799s, `.cpu` mean 0.012s, N=50); token counts were called out as
"track-only, no thresholds" in
[`downstream-validation/tasks.md` W0.4](../downstream-validation/tasks.md).
D5 produces the first *systematic* token-cost record — under the
current Anthropic SDK API this means parsing `response.usage` from
each rerank call.

### 1.3 Faithfulness vs retrieval-only

**Scoping decision required.** Two candidate scopes:

- **Scope α (retrieval-only).** D5 measures P@1, R@3, tokens, cites
  existing latency numbers. Verdict considers retrieval lift × cost.
  Faithfulness stays where it lives today
  ([`release-quality-baseline`](../release-quality-baseline/)).
- **Scope β (retrieval + faithfulness).** D5 also measures mean
  faithfulness under `rerank=off` vs `rerank=on`. Verdict considers
  retrieval + faithfulness lift × cost.

Scope α is the proposed default. Reasoning: faithfulness depends on
the *generation* step downstream of retrieval; if the reranker
changes which document ends up at rank 1, faithfulness deltas mostly
reflect "did the new top-1 document have the answer?" — which is
already what P@1 measures. Adding a second LLM-as-judge per query
(faithfulness scorer) ~doubles the measurement cost without obvious
new information.

Scope β is the scoping pass's call if the verdict needs to address
"the reranker may not lift retrieval but improves answers" — but
that's a hard story to argue against the current bundled corpus
where keyword retrieval already lands the right top-1 for 100% of
baseline queries.

**Recommendation for scoping:** Scope α, with Scope β reserved as a
follow-up if Scope α's verdict is "marginal retrieval lift, hold for
faithfulness data."

## 2. How to measure (candidate approaches)

### 2.1 Approach A: bolt onto existing test_golden.py runner

**Mechanism.** `tests/golden/test_golden.py` already runs the
40-baseline + 80-paraphrased set. Run it twice with `rerank=off` and
`rerank=on` (via `RagPipeline` construction-arg flip), capture
metrics + `response.usage` payloads, render a markdown report.

**Pros.**
- Reuses the exact pipeline construction the regression suite uses;
  zero risk of "diagnostic measures something different from what
  the pipeline ships."
- Minimal new code — a script that calls into the test fixtures.
- Same query set, same scoring code as the locked baseline numbers.

**Cons.**
- Test fixtures are not designed for output-capture. The runner may
  need a small refactor to expose per-query metrics + usage.
- Coupling: a future test-suite refactor could break the diagnostic.

### 2.2 Approach B: separate diagnostic script under `scripts/`

**Mechanism.** New `scripts/measure_reranker.py`, modeled on
[`scripts/measure_baseline_variance.py`](../../../scripts/measure_baseline_variance.py).
Constructs `RagPipeline` directly (twice — with and without
reranker), iterates the locked query YAMLs, writes the report.

**Pros.**
- Decoupled from the test suite; future refactors of either side
  don't break the other.
- Naturally produces a self-contained artifact (script → report).
- Same shape as the existing measurement-script pattern in
  `scripts/`.

**Cons.**
- Marginal duplication with the test runner — same query iteration,
  same scoring logic.

### 2.3 Approach C: extend `measure_baseline_variance.py` with a `--rerank-ablation` flag

**Mechanism.** Add a flag to the existing perf/quality variance
measurement script that runs each query both ways and produces a
diff column.

**Pros.**
- Single script for "all measurement-style work."
- Multi-run support inherited for free (could repeat the rerank
  ablation N times for variance characterization).

**Cons.**
- `measure_baseline_variance.py` is purpose-built for noise-floor
  measurement (N runs of the same configuration). Bolting an
  ablation onto it muddies its purpose. The two intents — "how noisy
  is this metric" vs "what's the lift from this component" — are
  orthogonal.

**Recommendation for scoping:** Approach B (separate script).
Decoupled, self-contained, matches the `scripts/` pattern. Approach
A is the fallback if scoping decides B duplicates too much logic.

## 3. Report shape

Mirrors [`baseline-1.md`](../release-quality-baseline/baseline-1.md)
where possible. Sketch:

```markdown
# Reranker evaluation (D5)

> Diagnostic report. Generated by `scripts/measure_reranker.py`.

| Field | Value |
|---|---|
| Measured at | `<ISO-8601>` |
| Commit | `<sha>` |
| Reranker model | `claude-haiku-4-5-20251001` |
| Baseline queries | `tests/golden/queries.yaml` (SHA `…`) |
| Paraphrased queries | `tests/golden/queries_paraphrased.yaml` (SHA `…`) |
| `candidate_multiplier` | 3 |
| Runs (N) | 1 (single-pass; see §5 for noise discussion) |

## Retrieval quality (the lift)

| Set | rerank=off P@1 | rerank=on P@1 | Δ P@1 | rerank=off R@3 | rerank=on R@3 | Δ R@3 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (40) | 1.000 | … | … | 1.000 | … | … |
| Paraphrased (80) | 0.9125 | … | … | 1.000 | … | … |
| Para — easy | … | … | … | … | … | … |
| Para — medium | … | … | … | … | … | … |
| Para — hard | … | … | … | … | … | … |

## Token cost

| Metric | Mean | p50 | p95 | Total / 80 queries |
|---|---:|---:|---:|---:|
| Input tokens | … | … | … | … |
| Output tokens | … | … | … | … |
| Cost (USD, Haiku list) | … | … | … | … |

## Latency (cited from `perf-baseline.md`)

| Metric | Mean | p95 | N |
|---|---:|---:|---:|
| `llm_reranker_rerank.wall` | 0.799 | 1.563 | 50 |
| `llm_reranker_rerank.cpu` | 0.012 | 0.071 | 50 |

(See `docs/specs/downstream-validation/perf-baseline.md`.)

## Residuals (paraphrased P@1 misses under rerank=off)

| Query ID | Expected top-1 | rerank=off top-1 | rerank=on top-1 | rerank fixes? |
|---|---|---|---|---|
| qp-… | concepts/… | tasks/… | … | y/n |
…

## Verdict

`<rerank-default-on | rerank-default-off | corpus-shape-dependent>`

<one-paragraph rationale.>

## Implications for user-corpus-onboarding

<one-paragraph: what the harness default should be; what the guide
should say about when to enable rerank on a user corpus.>
```

## 4. Verdict logic (proposed; scoping confirms)

Three candidate verdicts and the rough thresholds that map onto
each:

| Verdict | Trigger |
|---|---|
| `rerank-default-on` | Paraphrased P@1 lifts by ≥3 points (≥3 of 7 misses fixed) AND token cost is ≤ ~$0.001/query AND no regression on baseline. |
| `rerank-default-off` | Paraphrased P@1 lifts by ≤1 point OR baseline regresses OR token cost > ~$0.005/query. |
| `corpus-shape-dependent` | The verdict is genuinely "it depends" — i.e. the lift is real but token-cost-sensitive, and the right call is to expose the toggle prominently rather than pick a default. |

Thresholds above are illustrative — the scoping pass tightens them.
The point of the table is to commit to *deciding from the numbers*
rather than choosing a default based on intuition.

## 5. What D5 does not decide

D5 produces the verdict and the rationale. It does **not**:

- **Flip the pipeline default.** If verdict is `rerank-default-off`,
  the flip is a separate `### Changed` PR — possibly within Phase 4
  freeze (it's internal selection-criteria; not `### Added`), or
  possibly deferred to the v1.0.0 cut PR. Scoping picks.
- **Rewrite the reranker prompt.** Prompt-tuning is a separate
  follow-up spec if D5's verdict surfaces "marginal lift, prompt
  could be better."
- **Re-architect rerank candidate over-fetch.** `candidate_multiplier=3`
  is the current default; D5 measures *at that setting*. If the
  numbers suggest the multiplier is wrong, that's a separate spec.
- **Decide on cross-vendor rerankers.** Cohere, Voyage, BGE — out
  of scope per
  [README "What's not in scope"](README.md#whats-not-in-scope).

## 6. Single-run vs multi-run

The reranker is the only Anthropic-bound component on this path, so
its output is *non-deterministic* in a way keyword retrieval is not.
Two single-pass measurements may not produce identical numbers.

Three options:

- **Single-pass.** One run with `rerank=off`, one with `rerank=on`,
  report the numbers, note in the report that the rerank-on column
  is single-pass. Cheapest. Risk: noise on the 7 residual queries
  could move the verdict.
- **Small-N rerank-on.** Run `rerank=on` 5× and report
  mean ± stdev. Run `rerank=off` once (deterministic). Catches
  prompt-noise without ballooning the measurement budget — 5×
  ~80-query reruns × ~0.8s = ~5 minutes of wall-clock, ~400 Haiku
  calls.
- **Multi-run both columns.** Wasteful; the keyword-only column is
  deterministic.

**Recommendation for scoping:** small-N rerank-on (N=5). The 80-query
× 5-run × ~$0.0008 ≈ ~$0.30 of API spend; cheaper than the
ambiguity an unstable verdict would produce.

## 7. Interaction with other Phase 5 work

### 7.1 Sequencing vs `user-corpus-onboarding`

D5 **must land before user-corpus-onboarding M1 scoping closes** so
the harness default inherits D5's verdict. Per
[`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md#7-the-reranker-evaluation-diagnostic-d5-and-the-harness-happen-in-different-timeframes).

If both specs run concurrently in Phase 5, D5 should be M-numbered
to complete a week before user-corpus-onboarding M1's `/spec`-pass
scoping.

### 7.2 Sequencing vs `perf-baseline-multi-run`

D5 cites
[`perf-baseline.md`](../downstream-validation/perf-baseline.md) for
latency numbers. If `perf-baseline-multi-run` re-locks the baseline
mid-Phase-5, D5's report should re-cite from the post-lock numbers.
Otherwise no interaction.

### 7.3 Sequencing vs the v1.0.0 cut

D5's verdict shapes (possibly) one of two things in the cut:

- **A default-flip `### Changed` entry** in CHANGELOG (if verdict is
  `rerank-default-off`). This is selection-criteria; allowed under
  freeze.
- **A v1.0.0 POLICY.md commitment paragraph** about the reranker —
  whether the package commits to keeping it default-on, default-off,
  or exposes it as a corpus-shape-dependent decision.

Neither of those is forced by D5; both are decisions the cut PR's
own scoping pass takes, informed by D5's report.

## 8. Strict-dominance constraint

The diagnostic itself is read-only — it runs the existing pipeline
twice and reports numbers. It cannot, by construction, regress the
bundled corpus.

The **default flip** that D5 may trigger has strict-dominance
consequences:

- If verdict is `rerank-default-off` and the flip lands as
  `### Changed`, the flip PR must run the full baseline diagnostic
  and demonstrate that baseline P@1 / R@3 stay at 100% / 100% and
  paraphrased R@3 stays at 100% under the new default. If
  paraphrased P@1 drops, that's the expected delta and the PR body
  documents it.
- The flip PR is the strict-dominance gate; D5's report is the
  evidence the flip PR cites.

## 9. Open questions for scoping

(Each gets a decision at the `/spec` pass; listed here as the
unresolved set the scoping pass inherits.)

1. **Scope α vs β** (retrieval-only vs +faithfulness) — §1.3.
2. **Approach A / B / C** for the measurement implementation — §2.
3. **Single-pass vs N=5 rerank-on** — §6.
4. **Verdict thresholds.** §4's numbers are illustrative; pin them.
5. **Report location.** `docs/specs/reranker-evaluation/report.md`
   vs `docs/specs/reranker-evaluation/diagnostic-1.md` (the
   embedding-retriever convention).
6. **Default-flip PR timing.** If verdict is `rerank-default-off`,
   does the flip land within Phase 4 freeze (`### Changed`,
   selection-criteria-allowed) or at the v1.0.0 cut PR? Both legal;
   pick.
7. **Token-cost source.** `response.usage` from the Anthropic SDK,
   or compute from the request payload? SDK is simpler; payload
   compute is decoupled from SDK changes.
