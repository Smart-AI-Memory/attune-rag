# Spec: reranker-evaluation (attune-rag)

> **Status:** **complete 2026-05-22** — verdict locked at
> [`diagnostic-1.md`](diagnostic-1.md): **`rerank-default-off`**.
> D5 ran ahead of the 0.2.0 cut via the Phase 5 pre-staging arc.
> The verdict ratifies the existing `RagPipeline.reranker=None`
> default — no default-flip PR is required at the v1.0.0 cut.
> Cross-link landed: [`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md);
> follow-up note in [`v1.0.0-release/design.md`](../v1.0.0-release/design.md).

## Purpose

Measure whether [`LLMReranker`](../../../src/attune_rag/reranker.py)
earns its place as a default-on component of the v1.0.0 retrieval
pipeline.

`LLMReranker` (Claude Haiku) was wired into [`RagPipeline`](../../../src/attune_rag/pipeline.py)
during the early retrieval-quality work and has shipped enabled-by-default
through 0.1.x. The alias-expansion sweep (2026-05-21) then lifted the
**keyword-only** baseline to 100% / 100% on the bundled corpus and
paraphrased R@3 to 100% — meaning the reranker's *lift contribution*
on the bundled corpus is now structurally bounded: there's only the
paraphrased P@1 gap (91.25%, 7 of 80) left for it to close, and
non-trivial token + latency cost to weigh against any lift.

D5 produces a measurement report that answers, in numbers:

1. **Does the reranker lift P@1 / R@3** on the bundled corpus over
   keyword-only retrieval?
2. **At what token cost** (input + output, per query, aggregate)?
3. **At what latency cost** beyond what [`w2-perf-audit.md`](../downstream-validation/w2-perf-audit.md)
   already captured (~0.8 s wall, ~12 ms CPU)?
4. **What's the recommended v1.0.0 default** — `enabled`, `disabled`,
   or `corpus-shape-dependent`?

The verdict-and-default piece is what makes D5 load-bearing for the
v1.0.0 framing.

## Why now (Phase 5, not earlier or later)

**Earlier (e.g. mid-Phase-4):** premature. The alias-expansion sweep
re-set the keyword-only baseline mid-Phase-4; measuring reranker lift
*before* the sweep would have produced numbers that don't reflect the
pipeline shape v1.0.0 will actually ship. D5 needs the post-sweep
keyword floor as its comparison point.

**Later (post-v1.0.0):** too late. The
[`user-corpus-onboarding`](../user-corpus-onboarding/) harness ships
in Phase 5 with a default rerank posture (per its
[Risk §7](../user-corpus-onboarding/risks.md#7-the-reranker-evaluation-diagnostic-d5-and-the-harness-happen-in-different-timeframes)).
If D5 lands after the harness's M1 scoping, the harness default and
D5's verdict could disagree — two v1.0.0 surfaces saying different
things. Sequencing D5 *before* user-corpus-onboarding M1 lets the
harness inherit D5's recommendation.

**Now (Phase 5, item #1):** the
[Phase 5 scope decision](../v1.0.0-release/design.md#phase-5-scope-decided-2026-05-21)
records `perf-baseline-multi-run` and `user-corpus-onboarding` as the
two implementation tracks for Phase 5. D5 is an *upstream* diagnostic
that informs the harness default — small, read-only, runs ahead of
user-corpus-onboarding M1.

## What ships (the deliverable)

A single markdown report — `docs/specs/reranker-evaluation/report.md`
or similar (location pinned at scoping) — that mirrors the shape of
[`docs/specs/release-quality-baseline/baseline-1.md`](../release-quality-baseline/baseline-1.md).

The report contains:

- **Retrieval-quality table:** P@1 and R@3 on baseline + paraphrased
  query sets, in two columns — `rerank=off` (keyword-only) vs
  `rerank=on` (current pipeline default). Per-difficulty breakdown if
  the harness produces one.
- **Token-cost table:** mean / p50 / p95 input tokens, output tokens,
  total tokens per query. Aggregated cost-per-80-query-run.
- **Latency table:** cross-link to
  [`w2-perf-audit.md`](../downstream-validation/w2-perf-audit.md) for
  the wall-clock and CPU-time numbers already locked in
  [`perf-baseline.md`](../downstream-validation/perf-baseline.md).
  D5 doesn't re-measure latency; it cites.
- **Per-query residuals:** the 7 paraphrased queries where `rerank=off`
  misses P@1 — does `rerank=on` close any of them? Names the misses.
- **Verdict:** one of `{rerank-default-on, rerank-default-off,
  corpus-shape-dependent-default}` with a one-paragraph rationale.
- **Implications for `user-corpus-onboarding`:** what the harness
  default should be, what the guide should say about when to enable
  the reranker on a user corpus.

The report is **the artifact**. No code changes ship under D5 itself
unless the verdict triggers a default flip (and that flip lands as
its own PR with its own scoping pass — see
[design.md §5](design.md#5-what-d5-does-not-decide)).

## What's *not* in scope

- **Re-tuning the reranker prompt.** The prompt at
  [`reranker.py:14-30`](../../../src/attune_rag/reranker.py) has had
  multiple iterations; D5 measures the *current* prompt, not a
  re-tuned one. If the verdict is "marginal lift," prompt re-tuning
  is a v1.1.0+ follow-up spec.
- **Faithfulness measurement under rerank vs no-rerank.** D5
  measures retrieval-quality (P@1, R@3) and token cost only.
  Faithfulness deltas are tracked in
  [`release-quality-baseline`](../release-quality-baseline/);
  cross-cutting the two metrics here would balloon scope. See
  [design.md §1.3](design.md#13-faithfulness-vs-retrieval-only).
- **Other rerankers.** Cohere Rerank, Voyage, BGE — out of scope.
  The current adapter is Claude Haiku; D5 measures that adapter
  against keyword-only. Cross-vendor comparison is a different,
  larger spec.
- **User-corpus reranker measurement.** D5 measures the bundled
  corpus. The harness shipped under `user-corpus-onboarding` lets
  *users* measure their own corpora; D5's verdict informs the
  harness *default*, not its capability.
- **Decisions about embeddings or hybrid retrieval.** Embeddings are
  permanently scope-deferred (see
  [`embedding-retriever`](../embedding-retriever/#scope-of-the-defer)).
  D5 doesn't reopen that.

## Layout

- [`design.md`](design.md) — proposed measurement approach for each
  of the three deliverable pieces (quality, cost, latency), with
  candidate options and alternatives considered. Resolved at the
  `/spec` pass.
- [`requirements.md`](requirements.md) — invariants the diagnostic
  must satisfy (strict-dominance, freeze-window compatibility,
  reproducibility, output shape).
- [`risks.md`](risks.md) — measurement-validity risks, the
  default-flip-during-freeze risk, prompt-drift after measurement.

Now executable:

- [`tasks.md`](tasks.md) — M0 entry-gate verification, M1
  measurement script (`scripts/measure_reranker.py`), M2
  diagnostic run + report draft, M3 verdict + cross-link to
  `user-corpus-onboarding` M1. The seven scoping questions from
  [design.md §9](design.md#9-open-questions-for-scoping) +
  [requirements.md "Open questions"](requirements.md#open-questions-for-scoping)
  are answered at the top of `tasks.md`.

## Provenance

Identified as D5 (architectural diagnostic #5) in the 2026-05-21
planning conversation that locked the
[Phase 5 scope decision](../v1.0.0-release/design.md#phase-5-scope-decided-2026-05-21).
Surfaced explicitly in
[`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md#7-the-reranker-evaluation-diagnostic-d5-and-the-harness-happen-in-different-timeframes)
as the diagnostic the harness default depends on.

The measurement shape draws on:

- [`docs/specs/release-quality-baseline/baseline-1.md`](../release-quality-baseline/baseline-1.md)
  for retrieval-quality report shape.
- [`docs/specs/downstream-validation/w2-perf-audit.md`](../downstream-validation/w2-perf-audit.md)
  and
  [`perf-baseline.md`](../downstream-validation/perf-baseline.md)
  for the latency numbers D5 cites.
- [`docs/specs/embedding-retriever/diagnostic-1.md`](../embedding-retriever/diagnostic-1.md)
  for the format of "diagnostic report that produces a defer-or-ship
  verdict."
