# Diagnostic-2 — QueryExpander paraphrase robustness

**Status:** complete (informal diagnostic)
**Date:** 2026-05-21
**Owner:** Patrick
**Corpus:** `attune-help` v0.11.0
**Inputs:** [tests/golden/queries.yaml](../../../tests/golden/queries.yaml) (40 baseline), [tests/golden/queries_paraphrased.yaml](../../../tests/golden/queries_paraphrased.yaml) (80 paraphrases from D1)
**Driver:** [run_diagnostic_2.py](run_diagnostic_2.py)
**Raw output:** [diagnostic-2.run-output.md](diagnostic-2.run-output.md)
**Cache:** [diagnostic-2.cache.json](diagnostic-2.cache.json) (120 Haiku-expanded queries; reruns free)
**Summary stats:** [diagnostic-2.summary.json](diagnostic-2.summary.json)

## Question

D1 established that `KeywordRetriever` collapses on paraphrased queries (P@1 97.5% → 11.25%, R@3 100% → 28.75%). Before scoping an embedding retriever, would the **existing `QueryExpander`** (Haiku-backed query rewriter, already in the codebase, already wired into `RagPipeline._retrieve`) close most of that gap for free?

If yes, the embedding-retriever spec defers.

## Result

| | D1 (keyword only) | D2 (keyword + QueryExpander) | Δ vs D1 |
|---|---|---|---|
| **Baseline P@1** | 97.50% | 87.50% | **−10.00pp** ⚠️ |
| **Baseline R@3** | 100.00% | 97.50% | −2.50pp |
| **Paraphrased P@1** | 11.25% | 51.25% | **+40.00pp** |
| **Paraphrased R@3** | 28.75% | 80.00% | **+51.25pp** |

**Headline:** QueryExpander closes ~half the paraphrased P@1 gap and pushes R@3 from 28.75% to 80%. The verdict band is **STRONG** for the paraphrase case.

**Surprise (not predicted in the D2 plan):** the expander *regresses the baseline* by 10pp on P@1 and 2.5pp on R@3. On queries where keyword already had a clean unambiguous match, the expander's synonyms add tokens that pull adjacent features into the top-1 slot. **This means QueryExpander is not a free win** — it trades precision on keyword-friendly queries for recall on paraphrasic ones.

## Methodology

For each of the 40 baseline and 80 paraphrased queries, call `QueryExpander.expand(query)` to get 3–5 Haiku-generated alternative phrasings. Build the retrieval query as `original + " " + " ".join(expansions)` — exactly the way [pipeline.py:166-169](../../../src/attune_rag/pipeline.py:166) does it in production. Run `KeywordRetriever` against the expanded query. Score P@1 / R@3 per difficulty.

- 120 unique queries (no dedup overlap).
- ~200 output tokens per call, system-prompt cached.
- Sequential calls, 160.9s wall time, < $0.05 total cost.
- Expansions persisted to [diagnostic-2.cache.json](diagnostic-2.cache.json) so reruns are free and the result is reproducible against the same expansion artifact.

## Interpretation

### Why does the baseline regress?

QueryExpander is a *recall-boosting tool*. On a query like "fix failing tests" — already an exact-feature-name match for `concepts/tool-fix-test.md` — Haiku's expansions tend to drift toward adjacent concepts ("test coverage", "code quality issues", "CI debugging"). Those expansions add tokens that the keyword retriever then sees, which can:

1. Boost adjacent feature pages (e.g., `tool-smart-test.md`, `tool-code-quality.md`) enough to dethrone the correct match at rank 1.
2. Push the correct match out of the top-3 if the expansions are particularly noisy.

Both effects show up in the baseline numbers: 4 lost P@1 hits, 1 lost R@3 hit out of 40.

### Why does it work so well on paraphrased queries?

Paraphrased queries start with **zero token overlap** with the target by construction. Even an imprecise expansion that includes some near-target vocabulary is enough to surface the right template — the retrieval bar is "any signal at all" rather than "the strongest signal." That's why the +51pp R@3 lift is real even though Haiku's expansions are imperfect.

### Caveats

1. **Run-to-run variance unmeasured.** Haiku is stochastic. Single-run numbers are point estimates; the +40pp / +51pp result is likely within a few pp of the true mean but could swing on a re-run. To match D1's rigor, this needs N-run variance measurement before any production-defaulting decision.
2. **Cost is real.** 120 calls × ~$0.0004 = ~$0.05. Charged per user query at production rate. Defaulting QueryExpander on for all users with `[claude]` installed is not free, and the precision cost is not free either.
3. **D2 only measures retrieval.** Doesn't tell us whether faithfulness moves. Adjacent-feature noise in retrieved chunks could hurt or help generation quality.

## Combined picture with D1 + D3

| Lever | Paraphrase R@3 lift | Baseline cost | Dep cost | Per-query cost |
|---|---|---|---|---|
| Embedding retriever (proposed spec) | (unmeasured) | (unmeasured) | ~45 MB + new dep | embed-time + cosine compute |
| **QueryExpander (D2)** | **+51pp** | **−2.5pp R@3, −10pp P@1** | Haiku API | ~$0.0004 + latency |
| **Alias expansion (D3, bug-predict)** | **+50pp on cluster** | **0** (no regression observed) | **0** | **0** |

D3's lever is **strictly dominant** on every axis where it's been measured. D2 is a real second lever but with measurable costs. The embedding-retriever case has not held up against either dep-free alternative.

## Verdict

**STRONG** — QueryExpander closes most of the paraphrase gap.

**But: the right primary lever is D3 (alias expansion), not D2.** D3 has the same magnitude of recall lift on its cluster with zero baseline regression and zero dependency cost. D2 should be positioned as an opt-in capability for users whose query patterns trend paraphrasic, not as a default. And the embedding-retriever spec **defers** — there is no remaining justification for it at current corpus scale.

## Recommended next steps

1. **Defer the embedding-retriever spec.** Status banner across README/requirements/design/tasks → `deferred — D1+D2+D3 found cheaper alternatives`. Artifacts (paraphrase set, diagnostic scripts, summaries) stay on disk as the supporting evidence bundle.
2. **Open a smaller `alias-expansion-sweep` task** — generalize D3's lever across the other miss clusters (security-audit, refactor-plan, planning, code-quality, doc-orchestrator, doc-audit). Zero-dep, zero-baseline-regression, free to iterate.
3. **Decide on QueryExpander's user-facing positioning** as a *separate* small task — *do not* default it on. Document the precision tradeoff. Consider exposing the paraphrase benchmark in `--with-expander` mode in `benchmark.py` so users can see the tradeoff for their own query distribution before flipping it on.
4. **Promote `queries_paraphrased.yaml` to the regression suite** (info-only at first) regardless of which retriever path ships. The paraphrase signal is useful independent of how we close it.

## What this diagnostic does NOT decide

- Faithfulness with QueryExpander on. Separate diagnostic.
- N-run variance on the Haiku expansions. The +51pp / +40pp numbers are single-run point estimates.
- Whether `--with-expander` should gate CI. Likely no, given the variance and cost; info-only logging is the right initial posture.
- Whether the existing `[claude]` extra is the right home for QueryExpander long-term, or whether the expander itself should be provider-agnostic. Out of scope here.

## Files produced

- [run_diagnostic_2.py](run_diagnostic_2.py) — pure stdlib + PyYAML + anthropic SDK. Reusable with a cache file so reruns are free.
- [diagnostic-2.run-output.md](diagnostic-2.run-output.md) — captured run output, 2026-05-21.
- [diagnostic-2.cache.json](diagnostic-2.cache.json) — 120 cached Haiku expansions, idempotent reruns.
- [diagnostic-2.summary.json](diagnostic-2.summary.json) — machine-readable summary stats.
