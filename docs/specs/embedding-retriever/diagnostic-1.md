# Diagnostic-1 — KeywordRetriever paraphrase robustness

**Status:** scoping (informal diagnostic, no spec yet)
**Date:** 2026-05-21
**Owner:** Patrick
**Corpus:** `attune-help` v0.11.0
**Inputs:** [tests/golden/queries.yaml](../../../tests/golden/queries.yaml) (40 baseline), [tests/golden/queries_paraphrased.yaml](../../../tests/golden/queries_paraphrased.yaml) (80 paraphrases, 2 per baseline)
**Driver:** [run_diagnostic_1.py](run_diagnostic_1.py)
**Raw output:** [diagnostic-1.run-output.md](diagnostic-1.run-output.md)

## Question

Does `KeywordRetriever` degrade when queries are phrased without sharing tokens with the target template's path/summary/aliases? The size of that degradation is the EmbeddingRetriever case (or absence of one).

## Result

| Set | n | P@1 | R@3 |
|---|---|---|---|
| Baseline (original golden) | 40 | **97.50%** | **100.00%** |
| Paraphrased (no-overlap) | 80 | **11.25%** | **28.75%** |
| **Δ** | — | **−86.25pp** | **−71.25pp** |

By difficulty (paraphrased):

| difficulty | n | P@1 | R@3 |
|---|---|---|---|
| easy | 20 | 0.00% | 30.00% |
| medium | 54 | 14.81% | 29.63% |
| hard | 6 | 16.67% | 16.67% |

**Verdict:** **STRONG case for an embedding-based retriever** per the decision matrix (Δ P@1 > 15pp). 57 of 80 paraphrased queries (71%) fail R@3 — the keyword retriever doesn't just miss P@1, it misses the target template entirely most of the time.

## Methodology

For each of the 40 golden queries, two paraphrases were hand-authored under one constraint: **avoid tokens that appear in the target template's path / summary / aliases**, after the retriever's own stemming rules. Difficulty and `expected_in_top_3` were inherited unchanged.

The retriever was run as-is against both sets — no parameter sweeps, no reranker, no expander. The same `AttuneHelpCorpus.from_attune_help()` factory the production pipeline uses.

### Caveats (honest read)

1. **Some paraphrases are more idiomatic than developers would actually type.** Examples: gqp-006a *"where are the landmines in this commit"*, gqp-024a *"my suite is red — figure out why"*, gqp-036a *"where's the diff going to bite me"*, gqp-039a *"what should I tackle in the next two weeks"*. These contribute to the headline magnitude; a stricter "natural query log" sample would likely show a smaller gap.
2. **The magnitude is therefore inflated** — probably by 10–20pp. Even at the conservative end (say, −60pp on P@1), the verdict still maps to STRONG, so the decision is robust to the caveat.
3. **What this measures is purely retrieval** — not generation, not faithfulness, not end-to-end answer quality. A perfect embedding retriever doesn't automatically improve user-visible output.
4. **The diagnostic uses one embedding-favorable failure mode** (lexical mismatch). It doesn't simulate noisier failure modes (typos, multi-intent queries, ambiguous feature requests across two tools) where embeddings can also struggle.

## Secondary findings

- **Graceful degradation is poor.** When the keyword retriever doesn't find a strong feature match, it surfaces noise — `concepts/tool-attune-hub.md`, `concepts/socratic-discovery.md`, generic `tasks/task-configuration-setup.md` — rather than returning empty. One paraphrase (gqp-027b *"highlight worrisome spots in the diff"*) returned `[]`, which is arguably the correct failure mode but currently the exception.
- **`attune-hub` is a noise attractor.** It appears in the top-3 of 9 of 57 R@3 misses. Likely because its summary is general/meta (it routes between features), giving it broad token coverage. Worth a separate look orthogonal to the embedding question.
- **`task-*` templates outrank target `tool-*` templates on several paraphrases.** Continues the earlier `tasks/` vs `concepts/` ranking story (the 1.2× weight already applied; not enough headroom against lexical paraphrasing).
- **Difficulty axis is uninformative under paraphrase.** Easy queries drop to 0% P@1 — the baseline's easy/medium/hard split is keyword-difficulty, not semantic-difficulty.

## Decision matrix → next step

| Δ P@1 | Verdict | Action |
|---|---|---|
| ≤ 5pp | WEAK | Defer EmbeddingRetriever |
| 5–15pp | MIXED | Scope hybrid prototype |
| **> 15pp** | **STRONG** | **Scope full spec — local-model first, hybrid scoring, paraphrase benchmark gate** |

Observed −86pp (or −60pp+ even with the colloquial paraphrases stripped) → **STRONG**.

## Recommended next step

Open a real spec at `docs/specs/embedding-retriever/` (scoping → requirements → tasks). Initial design constraints to carry in:

1. **Local model first.** `sentence-transformers/all-MiniLM-L6-v2` or `fastembed` with a small ONNX model. No API dependency. Behind an `[embeddings]` extra so the base install stays light.
2. **Hybrid scoring, not replacement.** Blend `α · keyword + (1-α) · cosine`. Pure embedding loses on exact-feature-name queries; keyword wins those. Sweep α on a combined benchmark.
3. **Extend the golden set.** Promote `queries_paraphrased.yaml` to a permanent regression set (currently informal). Any retriever change must report against both. Decide whether paraphrased P@1 should gate CI or stay an info-only signal until variance is understood.
4. **Index lifecycle.** Embedding cache on `RetrievalEntry` (mirroring `_tokens_cache`) keyed by model name + chunking. Build-on-first-use is fine for current corpus size (~hundreds of docs); revisit if the corpus grows.
5. **Investigate the `attune-hub` noise attractor independently.** Doesn't depend on the embedding work and may close some of the gap on the keyword side for cheap.

## What this diagnostic does NOT decide

- Whether the hybrid α should be high (keyword-dominant) or low (embedding-dominant) — needs the prototype.
- Whether the existing baseline P@1 ≥ 0.95 gate should be raised, lowered, or supplemented with paraphrased-P@1.
- Whether faithfulness moves with retrieval quality. A separate diagnostic.
- Whether an OpenAI-hosted embedding API would meaningfully outperform a small local model on this corpus. Suspect not for ~hundreds of docs, but unmeasured.

## Files produced

- [tests/golden/queries_paraphrased.yaml](../../../tests/golden/queries_paraphrased.yaml) — 80 paraphrases, 2 per baseline query, same schema as `queries.yaml`.
- [run_diagnostic_1.py](run_diagnostic_1.py) — pure stdlib + PyYAML driver. Reusable for future "does retriever X handle paraphrasing" probes.
- [diagnostic-1.run-output.md](diagnostic-1.run-output.md) — captured run output for reproducibility.
