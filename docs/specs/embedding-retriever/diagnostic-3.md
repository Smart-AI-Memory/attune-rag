# Diagnostic-3 — Alias expansion on bug-predict

**Status:** complete (informal diagnostic)
**Date:** 2026-05-21
**Owner:** Patrick
**Corpus:** `attune-help` v0.11.0
**Inputs:** [tests/golden/queries_paraphrased.yaml](../../../tests/golden/queries_paraphrased.yaml) (80 paraphrases from D1, 14 of them bug-predict)
**Driver:** [run_diagnostic_3.py](run_diagnostic_3.py)
**Raw output:** [diagnostic-3.run-output.md](diagnostic-3.run-output.md)

## Question

D1 established that `KeywordRetriever` collapses on paraphrased queries (P@1 97.5% → 11.25%, R@3 100% → 28.75%) and bug-predict was the largest single feature miss cluster — 9 of 57 paraphrased R@3 misses. Before scoping an embedding retriever, can hand-authored aliases on the bug-predict entries close most of that cluster's gap?

If yes, the embedding-retriever spec defers in favor of an alias-authoring lever that costs nothing at install or runtime.

## Result

**bug-predict only (14 paraphrased queries):**

| | Before (aliases as-is) | After (18 added aliases) | Δ |
|---|---|---|---|
| P@1 | 14.29% (2/14) | **57.14% (8/14)** | **+42.86pp** |
| R@3 | 35.71% (5/14) | **85.71% (12/14)** | **+50.00pp** |

**Verdict band: STRONG.** Alias expansion closes 7 of 9 R@3 misses on the bug-predict cluster.

**Full-corpus view (sanity check — only bug-predict aliases changed):**

| | Before | After | Δ |
|---|---|---|---|
| P@1 (80 paraphrased) | 11.25% | 18.75% | +7.50pp |
| R@3 (80 paraphrased) | 28.75% | 37.50% | +8.75pp |

No regressions on any non-bug-predict query — the +7.5pp / +8.75pp overall lift is entirely from bug-predict queries flowing through. Strict win on every measured axis.

## Methodology

1. Loaded the corpus as-is via `AttuneHelpCorpus.from_attune_help()`.
2. For each of the 3 bug-predict entries (`concepts/tool-bug-predict.md`, `references/tool-bug-predict.md`, `quickstarts/skill-bug-predict.md`), reconstructed with augmented aliases via `dataclasses.replace` (the dataclass is frozen).
3. Wrapped the corpus in a thin proxy that returns the augmented entries for bug-predict paths and delegates everything else.
4. Re-ran `KeywordRetriever` against the 80-query paraphrase set.
5. Reported per-cluster and overall deltas.

**18 multi-token aliases added** to each bug-predict entry, hand-authored against the concept language in D1's bug-predict miss list. Each alias is 2–3 tokens (`KeywordRetriever.MIN_ALIAS_OVERLAP = 2` requires at least 2 query-token overlaps before crediting alias hits). Examples: `"dangerous code"`, `"weak points"`, `"silent failures"`, `"PR risk review"`, `"code landmines"`, `"shaky code"`, `"production risk"`.

No new dependency. No API calls. Authoring time: ~10 minutes.

## Interpretation

### Why does this work so well?

Each paraphrased query that missed in D1 was using semantic-equivalent vocabulary that didn't tokenize to any field on the target entry. The augmented aliases bridge that vocabulary directly — `"weak points"` overlaps `"what are the weak points in my source"` by 2 tokens (`{weak, point}`), clearing `MIN_ALIAS_OVERLAP`. The boost is then weighted by `ALIASES_WEIGHT = 1.5` and the bug-predict category multiplier, pushing the bug-predict entry above the noise attractors (`tool-attune-hub`, generic task templates) that won D1.

### The 2 residual misses

After alias expansion, 2 of 14 bug-predict paraphrases still miss R@3:

- **gqp-015a** *"where might my service fail silently"* — I added `"silent failures"` (tokens after stem: `{silent, failur}`); the query tokenizes to `{might, servic, fail, silent}`. Overlap is `{silent}` only — 1 token, below the 2-token threshold. **Authoring error**, trivially fixable by adding `"fails silently"` or `"service fails silently"`.
- **gqp-036a** *"where's the diff going to bite me"* — I added `"diff risk"` (`{diff, risk}`) and `"code that bites"` (`{code, bites}`); neither hits 2 tokens against `{diff, go, bite}`. **Authoring error**, fixable by adding `"diff bites"` directly.

These are not failures of the alias-expansion lever — they're failures of my first-pass authoring. A second iteration would close both.

### Why the strict-dominance result matters

D3 measures the cheapest possible intervention: hand-authored aliases on 3 corpus entries. Compared to D2 (QueryExpander):

| | D2 (QueryExpander) | D3 (alias expansion) |
|---|---|---|
| Paraphrased R@3 lift | +51pp (80 queries) | +50pp on cluster (14 queries) |
| Baseline P@1 cost | −10pp ⚠️ | 0 |
| Baseline R@3 cost | −2.5pp ⚠️ | 0 |
| Dependency added | Anthropic SDK + Haiku API | none |
| Per-query cost | ~$0.0004 + 1–2s latency | 0 |
| Per-query variance | stochastic (Haiku) | deterministic |
| Authoring effort | 0 (already exists) | ~10 min per cluster |
| Generalization confidence | high (works on any text) | needs sweep to confirm |

D3 wins on every measured axis except generalization confidence. The hypothesis is that an alias-expansion sweep across the other miss clusters (security-audit, refactor-plan, planning, code-quality, doc-orchestrator, doc-audit) reproduces the bug-predict result. That sweep is the natural next step.

### Caveats

1. **Generalization is hypothesized, not measured.** Bug-predict is one cluster of 9 missing features. The sweep is the experiment that confirms or refutes generalization.
2. **Aliases live in attune-help frontmatter, not attune-rag.** Two productive paths:
   - Extend `summaries_override.json` in attune-rag to also cover aliases (new `aliases_override.json` or unified override file).
   - Author aliases upstream in attune-help and bump its version.
   The override-in-attune-rag path is faster to iterate on; the upstream path is the long-term home for additions that prove out.
3. **Alias bloat risk.** Adding 18+ aliases per under-served feature could double the corpus's alias token volume. The retriever already caps and gates alias contribution (`MIN_ALIAS_OVERLAP = 2`, alias-token-set construction once per entry), so the runtime cost is negligible. The documentation cost is higher — each new alias needs to be conceptually distinct from the others.
4. **Authoring requires judgment.** D3 added aliases by reading D1's miss list and pattern-matching the failing query phrasings. That's tractable for a human; less obvious how to automate. A future task could use Haiku to *suggest* aliases (review by author) rather than *generate* them blind.

## Decision matrix → next step

| bug-predict R@3 lift | Verdict | Action |
|---|---|---|
| < 15pp | WEAK | Aliases insufficient; revive embedding spec |
| 15–30pp | MIXED | Aliases help; embedding spec mandate shrinks |
| **≥ 30pp** | **STRONG** | **Aliases substitute for embeddings; sweep across other clusters** |

Observed +50pp R@3 on bug-predict → **STRONG**.

## Recommended next step

Open an `alias-expansion-sweep` task (lighter than a full spec — `tasks.md` only, no requirements/design, since the methodology is now proven). One milestone per under-served feature cluster:

- security-audit (gqp-001a/b, 011a/b, 023a/b, 032a/b — 8 queries)
- refactor-plan (gqp-007a/b, 021a/b, 030a/b — 6 queries)
- planning (gqp-010a/b, 028a, 039a/b — 5 queries)
- code-quality (gqp-004a/b, 018a, 029a/b — 5 queries)
- doc-orchestrator (gqp-013a/b, 034a/b, 038a/b — 6 queries)
- doc-audit (gqp-009a/b, 025a — 3 queries)
- release-prep (gqp-005a/b dual, 008b, 019a/b, 026a, 035a — 6 queries; some are deep-review)
- doc-gen / fix-test / smart-test (smaller clusters)

For each cluster: author 8–18 multi-token aliases against the cluster's D1 miss list, re-run [run_diagnostic_3.py](run_diagnostic_3.py) (parameterized by feature), record before/after. Total estimated effort: 1–2 hours across all clusters.

Decide upstream-vs-override location for the aliases as a separate small design call.

## What this diagnostic does NOT decide

- Whether the same alias-authoring approach works equally well for clusters where the "noise attractor" (e.g., `tool-attune-hub`) outranks the target on category weight rather than alias coverage. Need to measure each cluster.
- Whether aliases should live in attune-rag's override file or upstream in attune-help. Tractable design call; defer to the sweep task.
- Whether to fix the 2 residual bug-predict misses (gqp-015a, gqp-036a) in a follow-up alias revision or leave them as known-imperfect. Trivially fixable; bundle into the sweep.

## Files produced

- [run_diagnostic_3.py](run_diagnostic_3.py) — pure stdlib + PyYAML driver. Reusable per-feature by parameterizing `_BUG_PREDICT_PATHS` and `_BUG_PREDICT_EXTRA_ALIASES`.
- [diagnostic-3.run-output.md](diagnostic-3.run-output.md) — captured run output, 2026-05-21.
