# Baseline with rerank — indicative measurement (N=1)

> **Scope.** This is the **N=1 indicative measurement** of rerank lift
> on the bundled `AttuneHelpCorpus` against the golden query sets, run
> 2026-05-22 via `python scripts/measure_corpus.py --with-rerank`. It
> is the *marketing artifact* — a real number to point at when telling
> users "spend ~$0.05 to see exactly where rerank lifts your corpus."
>
> **This is NOT the D5 verdict.** D5 ([`reranker-evaluation/`](../reranker-evaluation/))
> is the rigorous version: N=5 invocations, statistical CI on the
> default-flip decision, per-query stability annotation. The verdict
> rubric in [`reranker-evaluation/tasks.md` M3.1](../reranker-evaluation/tasks.md)
> is the load-bearing output — *not* this file. Treat the numbers
> below as a single non-stable sample, not a verdict.

## Run metadata

- Corpus: bundled `AttuneHelpCorpus` (attune-help templates)
- Baseline query set: `tests/golden/queries.yaml` (40 queries, sha256: `f47486df87c6`)
- Paraphrased query set: `tests/golden/queries_paraphrased.yaml` (80 queries, sha256: `307b6fcfa0d0`)
- Reranker: `claude-haiku-4-5-20251001` via `attune_rag.reranker.LLMReranker`
- Candidate multiplier: `3` (default)
- Runs: 1 (N=1; D5 uses N=5)
- Date: 2026-05-22
- Wall-clock: ~2 min 24 s
- API spend: under \$0.10 at Haiku list pricing (~120 rerank calls, ~300 tokens each)

## Aggregate result

| Set | n | Baseline P@1 | Baseline R@3 | +Rerank P@1 | +Rerank R@3 |
|-----|---|--------------|--------------|-------------|-------------|
| baseline (`queries.yaml`) | 40 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| paraphrased (`queries_paraphrased.yaml`) | 80 | 0.8750 | 0.9875 | 0.8750 | 0.9875 |

**Net rerank lift on aggregate: zero.** Same paraphrased P@1 (70/80)
and same R@3 (79/80) before and after rerank. The 10 paraphrased P@1
misses are robust to rerank — they reflect queries where the
keyword retriever doesn't surface an expected doc in the top-3
*candidate set*, so the reranker has nothing to lift from.

## Per-query: what actually moved

Across the 80 paraphrased queries, rerank changed the P@1 outcome on
exactly two queries — and the changes canceled out:

| qid | Without rerank | With rerank | Net |
|-----|----------------|-------------|-----|
| `gqp-003b` | ✗ | ✓ | +1 |
| `gqp-014b` | ✓ | ✗ | −1 |

R@3 unchanged on every query.

This is a single non-deterministic sample. With N=5 (D5's
methodology), we'd expect to see:
- A handful of additional queries with stability < 5/5 — rerank
  flipping inconsistently across runs
- Possibly a different net (e.g. ±1 or ±2) depending on which
  borderline candidates Haiku ranks first that day
- Aggregate variance bounded but non-zero

## What this tells us today (signal, not verdict)

1. **The bundled corpus is well-served by the keyword retriever.**
   The alias-expansion sweep closed the top-3 gap on every paraphrased
   query except one (`gqp-015b`-style hard queries). Rerank can't lift
   what's already at ceiling.

2. **Rerank's value is corpus-shape-dependent.** This corpus has
   curated multi-token aliases and a clean
   `aliases_override.json` — the keyword path already separates
   relevant from irrelevant in the top-3 set. On a less-aliased corpus
   (e.g. raw markdown without frontmatter discipline), rerank's lift
   would likely be material. The framework framing ([`v1.0.0-release`](../v1.0.0-release/))
   accounts for this: ship the *measurement tool*, let the corpus
   shape decide.

3. **N=1 is not the verdict.** D5 will measure N=5 and apply the
   rubric in `reranker-evaluation/tasks.md` M3.1. The thresholds:
   - `rerank-default-on` iff ≥3 of 7 paraphrased misses fix at ≥4/5
     stability AND token cost mean ≤ \$0.002/query
   - `rerank-default-off` iff ≤1 fix at ≥4/5 stability OR any
     baseline regression OR cost > \$0.005/query
   - `corpus-shape-dependent-default` iff exactly 2 fixes at ≥4/5

   Today's N=1 shows 1 fix and 1 regression on the same corpus —
   pointing toward `rerank-default-off` or `corpus-shape-dependent`,
   but a single sample can't distinguish.

## Use this artifact for

- **Marketing**: a concrete "spend ~\$0.05, get this data" example
  users can replicate on their corpus via
  `python scripts/measure_corpus.py --with-rerank ...`.
- **Onboarding**: the user-corpus guide
  ([`docs/USER_CORPUS_GUIDE.md`](../../USER_CORPUS_GUIDE.md) §6.2)
  can point here as the "what does the report look like in practice"
  example.

## Don't use this artifact for

- **The default-flip decision**. That's D5's job, with N=5 + the
  rubric. This file is N=1 and gets *replaced* (not amended) by
  `docs/specs/reranker-evaluation/diagnostic-1.md` once M2 runs.
