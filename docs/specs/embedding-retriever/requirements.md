# Spec: embedding-retriever — Requirements

> **Status: deferred — D1+D2+D3 found dependency-free alternatives close the gap; spec does not activate unless an alias-expansion sweep leaves a meaningful residual. See [README.md](README.md) for the full defer rationale.**

This spec adds a semantic retrieval path (local-model embeddings + hybrid combiner) to address the lexical-paraphrase brittleness measured in [diagnostic-1.md](diagnostic-1.md). The default retriever does not change.

## Phase 1: Requirements

**Status:** scoping — to be finalized by the `/spec` scoping pass.

### Entry gates

All gates below must be **green** before this spec promotes from "scoping" to "approved." Unlike `api-v0.2.0-cut`, the gates here are *behind* us — listed for traceability.

- [x] **Diagnostic-1 verdict is STRONG.** Δ P@1 > 15pp. Observed: −86.25pp. See [diagnostic-1.md §Result](diagnostic-1.md#result).
- [x] **Diagnostic-1 reproducible.** [run_diagnostic_1.py](run_diagnostic_1.py) is pure stdlib + PyYAML, takes < 5s, and the run output is captured at [diagnostic-1.run-output.md](diagnostic-1.run-output.md).
- [x] **Paraphrase set authored under a documented constraint.** [tests/golden/queries_paraphrased.yaml](../../../tests/golden/queries_paraphrased.yaml) — 80 entries, 2 per baseline query, no-token-overlap rule.
- [ ] **0.2.0 cut has landed** (per [api-v0.2.0-cut/](../api-v0.2.0-cut/)). This spec targets **0.3.0** as the additive minor; we don't open it inside the 0.2.0 freeze window. *(Gate may be relaxed during scoping if the team decides additive-only surface is safe during the freeze.)*

### Functional requirements (to be expanded during scoping)

#### F1 — `EmbeddingRetriever` exists and satisfies `RetrieverProtocol`

- [ ] New class `EmbeddingRetriever` in [`src/attune_rag/retrieval.py`](../../../src/attune_rag/retrieval.py) or a sibling module.
- [ ] Satisfies the existing `RetrieverProtocol` ([retrieval.py:147](../../../src/attune_rag/retrieval.py:147)) without modification to the protocol itself.
- [ ] Drop-in compatible with `RagPipeline(retriever=...)` per [pipeline.py:108](../../../src/attune_rag/pipeline.py:108).
- [ ] Operates on the same `RetrievalEntry` shape; no changes to `CorpusProtocol`.

#### F2 — Local-model only, behind an opt-in extra

- [ ] Embedding model is a small local model loaded once per process. Provisional choice: `sentence-transformers/all-MiniLM-L6-v2` (~80 MB) or `fastembed` ONNX equivalent. Final choice deferred to scoping.
- [ ] Added under a new `[embeddings]` extra in [pyproject.toml](../../../pyproject.toml). Base install footprint unchanged.
- [ ] Import is lazy — `attune-rag` imports without the extra; `EmbeddingRetriever()` raises a clear "install attune-rag[embeddings]" error when the extra is missing. Pattern mirrors how `attune-help` is handled by the pipeline today.
- [ ] No network calls at retriever construction or query time once the model is on disk.

#### F3 — `HybridRetriever` blends keyword + embedding

- [ ] New class `HybridRetriever` in the same module.
- [ ] Convex combination: `α · normalize(keyword_score) + (1 − α) · cosine_score`, with both component scores normalized to `[0, 1]` per query.
- [ ] `α` is a class attribute, sweepable by the benchmark (mirrors `KeywordRetriever.PATH_WEIGHT` style).
- [ ] Returns `RetrievalHit` with a `match_reason` of the form `hybrid:kw=0.43+emb=0.91 → 0.74` so the dashboard's reason column stays informative.

#### F4 — Per-entry embedding cache

- [ ] `RetrievalEntry._embeddings_cache` sidecar mirrors `_tokens_cache`. Key shape: `(model_name, content_preview_chars)`.
- [ ] Embedding cache survives across queries within a process; rebuilds on corpus rebuild (same behavior as `_tokens_cache`).
- [ ] First-query latency overhead documented in benchmark output (build-on-first-use is acceptable).

#### F5 — Paraphrase set joins the regression suite

- [ ] [tests/golden/queries_paraphrased.yaml](../../../tests/golden/queries_paraphrased.yaml) is loaded alongside `queries.yaml` by `tests/golden/test_golden.py` and by `src/attune_rag/benchmark.py`.
- [ ] Initial gating posture: **info-only signal** for one minor (0.3.x), then **gating threshold set** at 0.4.0 based on observed variance. Scoping decides whether to bring this forward.
- [ ] No changes to the existing P@1 ≥ 0.95 / R@3 ≥ 1.00 / faithfulness ≥ 0.9686 thresholds in this spec.

### Success criteria (to be made measurable during scoping)

The spec ships when:

- [ ] On the baseline `queries.yaml`: `HybridRetriever` matches `KeywordRetriever` on P@1 and R@3 within noise (no regression on the exact-feature-name queries that keyword handles perfectly today).
- [ ] On the paraphrased `queries_paraphrased.yaml`: `HybridRetriever` improves P@1 by at least **30pp absolute** over `KeywordRetriever` (i.e., 11.25% → ≥ 41.25%). Target band to be tightened during scoping after the prototype runs.
- [ ] Faithfulness on the baseline does not degrade (re-run the existing faithfulness benchmark with `HybridRetriever` plugged in; verify within the locked threshold).
- [ ] Total `pip install attune-rag[embeddings]` wheel + on-disk model footprint < **150 MB**.
- [ ] First-query latency overhead (model load + corpus embed) < **5 s** for the current `attune-help` corpus on a developer laptop. Steady-state per-query overhead < **50 ms**.

### Out of scope

- API-backed embedding providers (OpenAI, Voyage, Anthropic). Follow-up spec.
- Vector stores. In-memory cosine is sufficient at current corpus size; revisit at 10k+ docs.
- Chunking beyond one-vector-per-entry. Follow-up if the paraphrase benchmark plateaus below the success threshold.
- Default-retriever switch. `KeywordRetriever` stays the default through 0.3.x; the flip is a separate decision tied to evidence from the in-the-wild 0.3.x usage.
- The `attune-hub` noise-attractor finding from diagnostic-1. Orthogonal to embeddings; tracked separately.

### Open questions (for scoping)

To be answered by the `/spec` scoping pass:

1. **Local model choice.** `sentence-transformers/all-MiniLM-L6-v2` vs `fastembed` (BAAI/bge-small-en or similar). Tradeoffs: install footprint, license, cold-start latency, corpus-specific quality.
2. **Index lifecycle.** Build on first query (lazy) vs build on corpus construction (eager). Lazy is friendlier for `pip install` smoke tests; eager simplifies benchmarking. Provisional: lazy with explicit `build()` method for benchmarks.
3. **α default.** Start at 0.5 and sweep, or pick a principled starting point (e.g., reciprocal-rank-fusion-equivalent α). Sweep is cheap; just pick after the prototype runs.
4. **Score normalization shape.** Min-max within retrieved set, softmax, or rank-based. Rank fusion has the cleanest properties but loses score magnitude. Decide after the prototype.
5. **Should paraphrased-P@1 gate CI from 0.3.0, or wait?** Recommend: info-only at 0.3.0 (one minor of observed variance), gate at 0.4.0. Decide during scoping.
6. **MIN_SCORE behavior in hybrid.** `KeywordRetriever.MIN_SCORE = 2.0` filters out low-confidence keyword matches. The hybrid needs an analogous threshold that doesn't accidentally strip valid embedding-only matches. Provisional: gate only on the *blended* score, not the components.
7. **Should `HybridRetriever` be the recommended default for new users at 0.3.0 docs even if `KeywordRetriever` stays the code default?** Documentation lead vs code lead can diverge.
