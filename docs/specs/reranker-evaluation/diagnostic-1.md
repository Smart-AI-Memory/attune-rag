# D5 diagnostic: LLMReranker evaluation

## Reproducibility metadata

- anthropic_sdk_version: `0.104.0`
- baseline_queries_sha: `f47486df87c6`
- candidate_multiplier: `3`
- commit_sha: `5022f3d6b7a53577b5e892896c58ad025ea32605`
- harness_version: `0.1.0`
- paraphrased_queries_sha: `307b6fcfa0d0`
- rerank_runs: `5`
- reranker_model: `claude-haiku-4-5-20251001`
- run_b_skipped: `false`
- timestamp: `2026-05-22T00:00:00Z`

## Run A — `rerank=off` (deterministic, N=1)

| Set | P@1 | R@3 |
|-----|-----|-----|
| baseline | 1.0000 | 1.0000 |
| paraphrased | 0.8750 | 0.9875 |

**R1 strict-dominance check:** R1 reproduction OK: {'baseline_p1': 1.0, 'baseline_r3': 1.0, 'paraphrased_p1': 0.875, 'paraphrased_r3': 0.9875}

## Run B — `rerank=on` (Haiku, N=5)

| Metric | mean | p50 | p95 |
|--------|------|-----|-----|
| baseline_p1 | 0.9850 | 0.9750 | 1.0000 |
| baseline_r3 | 0.9950 | 1.0000 | 1.0000 |
| paraphrased_p1 | 0.8800 | 0.8875 | 0.8875 |
| paraphrased_r3 | 0.9825 | 0.9875 | 0.9875 |

## Per-query residuals (paraphrased P@1 misses in Run A)

Stability column is `k/N` = number of Run-B passes where rerank lifted that query to ✓. The rubric's "stable lift" bar is `≥ ⌈0.8·N⌉` (e.g. ≥4/5).

| qid | difficulty | rerank fixes P@1 (k/N) | R@3 stability | stable lift? |
|-----|-----------|------------------------|---------------|--------------|
| `gqp-001b` | easy | 0/5 | 5/5 | ✗ |
| `gqp-003b` | easy | 1/5 | 5/5 | ✗ |
| `gqp-005a` | easy | 0/5 | 3/5 | ✗ |
| `gqp-011b` | medium | 2/5 | 5/5 | ✗ |
| `gqp-015b` | medium | 0/5 | 5/5 | ✗ |
| `gqp-016b` | easy | 0/5 | 5/5 | ✗ |
| `gqp-017b` | easy | 2/5 | 5/5 | ✗ |
| `gqp-021b` | hard | 0/5 | 5/5 | ✗ |
| `gqp-023a` | medium | 0/5 | 0/5 | ✗ |
| `gqp-031a` | medium | 5/5 | 5/5 | ✓ |

**Stable-lift count:** 1 of 10 residuals fixed at ≥⌈0.8·N⌉ stability.

## Verdict

**`rerank-default-off`** — multiple independent triggers fire from
[`tasks.md` M3.1](tasks.md):

1. **Stable-lift bar:** ≤1 of the paraphrased P@1 misses fixed at
   ≥⌈0.8·N⌉ stability. We saw exactly 1 of 10 — only `gqp-031a`
   lifted 5/5. The rubric calls this a "rerank-default-off" trigger
   on its own.
2. **Baseline regression (load-bearing):** Run B baseline P@1 mean
   dropped from 1.00 to 0.985 — rerank moves a winning doc off
   rank 1 in roughly 1.5 % of baseline runs. Baseline R@3 also dropped
   below 1.00 (0.995). Paraphrased R@3 dropped from 0.9875 to 0.9825.
   Three separate regression conditions; any one alone is a
   "rerank-default-off" trigger per the rubric.
3. **Token cost:** Below the rubric's >$0.005/query threshold —
   ~600 calls in ~10 min wall-clock at Haiku list ≈ $0.50 total
   (~$0.00083/query). Cost is *not* the binding constraint here;
   the quality regression is.

**Story.** On this well-aliased corpus, the keyword retriever already
separates relevant from irrelevant in the top-3 candidate set. Rerank
has no headroom to lift winners, but it *does* have non-zero
probability of demoting them — which is exactly what the baseline
regression numbers capture. The single stable lift on paraphrased
(`gqp-031a`) doesn't compensate for the baseline noise.

**Implication for `RagPipeline.reranker` default.** The
[`pipeline.py`](../../../src/attune_rag/pipeline.py) default is
already `None` (off). **D5 ratifies the existing default.** No flip
is required — the verdict aligns with shipped behavior.

This is a *spec correction*: the
[`user-corpus-onboarding`](../user-corpus-onboarding/) scoping decision
#7 claimed "Mirror RagPipeline default (currently `on`)". That was
wrong about the current default. D5's verdict + the actual default
agree on `off`; the user-corpus-onboarding spec's Q7 reading is
corrected in this PR.

## Implications for `user-corpus-onboarding`

`scripts/measure_corpus.py`'s `--with-rerank` is the right shape: an
opt-in flag, not the default. Per the rubric the harness default
inherits this verdict — off — which is already what `measure_corpus.py`
ships.

`USER_CORPUS_GUIDE.md` §6.2's reframing (PR #134) aligned with this
verdict ahead of the run: *"measure whether rerank earns its keep"*
treats both lift and neutral as informative. The verdict ratifies that
framing as the correct shape: **run the measurement; let the corpus
shape decide.** On corpora without curated frontmatter aliases the
answer may flip, and that's fine — the framework gives them the tool
to see.

Cross-link: [`user-corpus-onboarding/risks.md` §7](../user-corpus-onboarding/risks.md)
updated in this PR with the verdict + harness-default decision.
