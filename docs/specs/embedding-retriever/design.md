# Spec: embedding-retriever — Design

> **Status: deferred (permanent as of 2026-05-21) — alias-expansion-sweep landed paraphrased R@3 = 96.25% with baseline 100%/100%. Design preserved as archival reference for the revival path (which now requires evidence the alias mechanism can't close a gap that matters for shipped usage). See [README.md](README.md) for the full rationale.**

This document sketches the shape of the embedding retriever and hybrid combiner. It is **not** a finalized design — the `/spec` scoping pass picks the model, fills in the algorithm parameters, and decides the CI-gate timing.

## Phase 2: Design

### Why now, and why this shape

Diagnostic-1 measured `KeywordRetriever` P@1 collapsing from **97.5% → 11.25%** on hand-authored, no-token-overlap paraphrases of the existing golden set. R@3 dropped from **100% → 28.75%**. The signal is unambiguous in direction; the magnitude is likely inflated 10–20pp by some idiomatic paraphrases, but the verdict band (>15pp) holds at the conservative end.

This justifies adding a semantic retrieval path. It does **not** justify replacing keyword retrieval, which is fast, deterministic, free, and wins on exact-feature-name queries. The right move is therefore an **additive** new retriever plus a **hybrid combiner** that the user (or future docs default) can opt into.

### Module layout

```
src/attune_rag/
  retrieval.py            # KeywordRetriever (unchanged), RetrieverProtocol (unchanged)
  retrieval_embedding.py  # NEW — EmbeddingRetriever, HybridRetriever
                          #       (separate file so `import attune_rag.retrieval`
                          #       stays free of the optional embedding dep)
  corpus/base.py          # RetrievalEntry — adds _embeddings_cache sidecar
                          #       (parallel to _tokens_cache)
```

Both new classes live in a sibling module rather than `retrieval.py` itself so the existing module's import surface stays dep-free. `retrieval_embedding.py` does the optional-dep import at module top with a clear `ImportError` message that points at `pip install attune-rag[embeddings]`.

### `EmbeddingRetriever` algorithm

```
Input:  query (str), corpus, k
Output: list[RetrievalHit] sorted by descending cosine similarity

1. q_vec = model.encode(query)
2. For each entry in corpus.entries():
   - If entry has no cached embedding for this model:
     - text = f"{entry.path} {entry.summary} {entry.content[:N]}"
     - entry._embeddings_cache[(model_name, N)] = model.encode(text)
   - score = cosine(q_vec, cached)
3. Filter: score >= MIN_COSINE  (default 0.20, configurable)
4. Sort by descending score, tie-break on entry.path for determinism
5. Return top-k as RetrievalHit(entry, score, match_reason="embed:0.87")
```

**Design notes:**

- The embed-text concatenation includes `path` because filename tokens carry signal that summaries sometimes miss (e.g., `tool-bug-predict.md`). The model handles the noise from path punctuation natively.
- `N = CONTENT_PREVIEW_CHARS = 500` mirrors the keyword retriever's preview window. Diverging would mean two cached states per entry.
- Aliases are concatenated into the embed text — they carry explicit synonym signal that the model would otherwise have to infer.
- `MIN_COSINE` filtering is conservative; for a small corpus, the top-k is rarely below 0.2 unless the query is genuinely off-topic. Tune during prototype.

### `HybridRetriever` algorithm

```
Input:  query (str), corpus, k
Output: list[RetrievalHit] sorted by descending blended score

1. kw_hits = KeywordRetriever(...).retrieve(query, corpus, k=K_INNER)
2. emb_hits = EmbeddingRetriever(...).retrieve(query, corpus, k=K_INNER)
   # K_INNER = 10 (wider net than the user's k, so the blend has options)
3. Union the candidate set: { entry for hit in kw_hits ∪ emb_hits }
4. For each candidate entry, compute:
   - kw_score_norm  = kw_hit.score / max(kw_hit.score for hit in kw_hits)   (0 if not in kw_hits)
   - emb_score_norm = emb_hit.score / max(emb_hit.score for hit in emb_hits) (0 if not in emb_hits)
   - blended = α · kw_score_norm + (1 − α) · emb_score_norm
5. Filter: blended >= MIN_BLENDED  (default 0.10, configurable)
6. Sort by descending blended, tie-break on entry.path
7. Return top-k as RetrievalHit(entry, blended, match_reason="hybrid:kw=...+emb=...")
```

**Why this shape:**

- **Per-query normalization** before blending — raw keyword scores can be >50 while cosine is in `[0, 1]`. Without normalization, α stops being interpretable. Min-max across the inner retrieved set is the simplest defensible choice; rank-based fusion (RRF) is a follow-up if the convex combination underperforms.
- **Wider inner k** so the blend can rescue entries that one retriever ranked outside the user's top-k. `K_INNER = 10` is provisional.
- **Filter on blended, not components.** Diagnostic-1 showed the keyword retriever returning `[]` once (the empty filter), and noise top-3 in many other cases. The hybrid should gracefully degrade to no-match rather than emit noise; a blended-threshold filter is the right level.

### Cache shape on `RetrievalEntry`

Today, [base.py](../../../src/attune_rag/corpus/base.py) entries carry a `_tokens_cache: dict` keyed by tuples like `("field_tokens", CONTENT_PREVIEW_CHARS)`. The embedding cache uses the same dict (a single sidecar attribute) but with non-overlapping keys:

```python
entry._tokens_cache[("field_tokens", 500)]                 = {...token sets...}
entry._tokens_cache[("embedding", "all-MiniLM-L6-v2", 500)] = np.ndarray  # 384-dim
```

Naming: the dict is `_tokens_cache` for historical reasons; an embedding key inside it is awkward. Provisional plan: rename the sidecar to `_cache: dict` at the same minor (additive, no public surface impact) and let both retrievers store under it. Final naming decision deferred to scoping.

### `[embeddings]` extra and dep choice

| Candidate | Wheel size | Disk size | Cold start | License | Notes |
|---|---|---|---|---|---|
| `sentence-transformers` + `all-MiniLM-L6-v2` | ~7 MB (pkg) + torch | 80 MB model | ~2 s | Apache-2.0 | Brings `torch` (~600 MB on most platforms) as a transitive dep. Big footprint. |
| `fastembed` + BGE-small | ~10 MB | 35 MB model | <1 s | Apache-2.0 (lib), MIT (model) | ONNX runtime, no torch. Significantly lighter. **Provisional pick.** |
| `model2vec` static embeddings | ~5 MB | 8 MB | <100 ms | Apache-2.0 | Static word-vector-style; quality likely lower than transformer-based. Worth measuring. |

`fastembed` is the provisional pick on footprint + cold-start grounds. The `/spec` scoping pass runs a small bake-off (run diagnostic-1 against each, compare P@1 deltas) before locking in.

The extra is wired in [pyproject.toml](../../../pyproject.toml):

```toml
[project.optional-dependencies]
embeddings = ["fastembed>=0.3,<1.0"]   # or sentence-transformers, TBD
```

### Benchmark integration

[`src/attune_rag/benchmark.py`](../../../src/attune_rag/benchmark.py) currently runs against `tests/golden/queries.yaml`. Changes:

- New `--queries-paraphrased PATH` flag (defaults to `tests/golden/queries_paraphrased.yaml`).
- New `--retriever {keyword,embedding,hybrid}` flag (defaults to `keyword`).
- Output adds a separate section per query set so we can read baseline and paraphrased side by side.
- α sweep is a new mode (`--alpha-sweep 0.0,0.25,0.5,0.75,1.0`) that runs the hybrid at each α and prints the matrix.

### What 0.3.0 means for this spec

| Aspect | 0.2.x | 0.3.0 (this spec lands) |
|---|---|---|
| Default retriever | `KeywordRetriever` | unchanged |
| `EmbeddingRetriever` public symbol | absent | added to `__all__` under deprecation policy §4 (add procedure) |
| `HybridRetriever` public symbol | absent | added to `__all__` under §4 |
| `[embeddings]` extra | absent | added |
| `tests/golden/queries_paraphrased.yaml` | informal (this spec's diagnostic) | promoted to regression set |
| Paraphrased-P@1 gate | none | info-only signal in CI |
| Faithfulness threshold | 0.9686 | unchanged |
| `KeywordRetriever` API | stable | unchanged |
| Existing P@1 / R@3 thresholds on baseline | 0.95 / 1.00 | unchanged |

**This spec adds surface; it does not remove or rename anything.** Compatible with the 0.2.0 SemVer commitment.

### Risks & mitigations (sketch)

To be expanded during scoping. Initial candidates:

| Risk | Mitigation sketch |
|---|---|
| Embedding model adds enough install footprint that casual users abandon. | `[embeddings]` is opt-in; base install size unchanged. Pick `fastembed` over torch-based options. Document size up-front in the README. |
| Embedding retriever underperforms keyword on exact-feature-name queries (the case where keyword is at 100% today). | Hybrid by default for users who enable embeddings; α tuned so keyword dominates on high-keyword-overlap queries. Baseline regression test fails the change if pure-keyword behavior degrades. |
| Paraphrase set has authorship bias (one author, idiomatic phrasing). | Documented in [diagnostic-1.md §Caveats](diagnostic-1.md#caveats-honest-read). Verdict is robust to the bias at 60pp margin. Future: solicit paraphrases from second author or from real query logs once usage data exists. |
| Cold-start latency surprises CI or first-use. | Lazy index build on first query is the default; explicit `EmbeddingRetriever.build_index(corpus)` for benchmarks. First-query overhead measured and reported in benchmark output. |
| Embedding quality on the `attune-help` corpus is worse than expected (small docs, dense jargon). | Bake-off in M1 runs all candidate models against diagnostic-1's paraphrase set before picking. If no model clears the success criterion, we report negative result and defer rather than ship a worse retriever. |
| `model2vec` or another static-embedding option is "good enough" and we over-engineered. | M1 includes a static-embedding baseline so we know what the floor looks like. |
| Future: API-backed embeddings (OpenAI, Voyage) make the local-model choice obsolete. | The `EmbeddingRetriever` interface accepts any `encode(text) -> vector` callable. API providers slot in as a follow-up spec without rework. |
| Paraphrase set itself becomes outdated as the corpus evolves. | The diagnostic script is committed and reusable; re-run is a single command. Add a check to `[Unreleased]` review process that flags when `attune-help` releases change template summaries materially. |

### Open design questions

To be resolved by the `/spec` scoping pass — see [requirements.md §Open questions](requirements.md#open-questions-for-scoping). Most consequential ones:

1. **fastembed vs sentence-transformers vs model2vec.** Bake-off in M1 picks.
2. **Sidecar rename `_tokens_cache` → `_cache`.** Cosmetic but affects both retrievers.
3. **α sweep methodology.** Pick a default from the sweep; document the chosen value as a class attribute.
4. **When does the paraphrased-P@1 gate become a hard CI gate?** Recommend 0.4.0; scope confirms.
