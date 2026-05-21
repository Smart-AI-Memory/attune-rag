# Spec: embedding-retriever — Tasks

> **Status: deferred (permanent as of 2026-05-21) — not executable. alias-expansion-sweep landed paraphrased R@3 = 96.25% with baseline 100%/100%, exceeding the M13.2 revival threshold. See [README.md](README.md) for the full rationale.**

This file is a **scaffold**, not a runnable task list. A `tasks.md` in a scoping spec is **not executable** — see user memory `feedback_spec_scoping_vs_approved`. The `/spec` pass promotes this into an approved spec with concrete scripts, acceptance criteria, and dependency arrows.

## Phase 3: Tasks

**Status:** scoping. No work in this spec is approved to start.

### Implementation order (sketch)

Four milestones. M1 is the bake-off that picks the model; M2 builds the hybrid; M3 wires the benchmark and adds the paraphrase regression; M4 cuts 0.3.0.

| # | Task | Layer | Status | Notes (to be filled by scoping) |
|---|------|-------|--------|---------------------------------|
| **M1 — Model bake-off and `EmbeddingRetriever` skeleton** | | | | |
| M1.1 | Add `[embeddings]` extra to `pyproject.toml` (provisional: `fastembed`). Verify base install still works without it. | attune-rag | scaffold | unscoped |
| M1.2 | Write `EmbeddingRetriever` skeleton in `src/attune_rag/retrieval_embedding.py` satisfying `RetrieverProtocol`. Lazy index build. Optional-dep import guarded with a clear `ImportError`. | attune-rag | scaffold | unscoped |
| M1.3 | Extend `RetrievalEntry` sidecar to hold embeddings (decide rename `_tokens_cache → _cache` here). | attune-rag | scaffold | unscoped |
| M1.4 | Run [run_diagnostic_1.py](run_diagnostic_1.py) (extended to accept a retriever flag) against three candidates: `fastembed/bge-small-en`, `sentence-transformers/all-MiniLM-L6-v2`, `model2vec/potion-base-2M`. Compare paraphrased P@1, install footprint, first-query latency. **Pick one.** Document the verdict in a new `model-bakeoff.md` alongside diagnostic-1.md. | attune-rag | scaffold | unscoped |
| **M2 — `HybridRetriever`** | | | | |
| M2.1 | Implement `HybridRetriever` in `retrieval_embedding.py` per [design.md §HybridRetriever algorithm](design.md#hybridretriever-algorithm). | attune-rag | scaffold | unscoped |
| M2.2 | α sweep mode added to the benchmark; run `α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` against both baseline and paraphrased query sets. Pick the α that maximizes (baseline P@1) subject to (paraphrased P@1 ≥ baseline keyword P@1 + 30pp). Document in `alpha-sweep.md`. | attune-rag | scaffold | unscoped |
| M2.3 | Set `HybridRetriever.ALPHA = <picked>` as a class attribute; document the rationale and the sweep table inline. | attune-rag | scaffold | unscoped |
| **M3 — Benchmark integration and paraphrase regression** | | | | |
| M3.1 | Add `--queries-paraphrased PATH` and `--retriever {keyword,embedding,hybrid}` flags to `src/attune_rag/benchmark.py`. | attune-rag | scaffold | unscoped |
| M3.2 | Promote `tests/golden/queries_paraphrased.yaml` to a regression input in `tests/golden/test_golden.py` — info-only assertions for 0.3.x (log P@1 / R@3 without failing). | attune-rag | scaffold | unscoped |
| M3.3 | Re-run the faithfulness benchmark with `HybridRetriever` plugged in (default α). Verify mean faithfulness stays within the locked threshold (0.9686). Record numbers in `faithfulness-with-hybrid.md`. | attune-rag | scaffold | unscoped |
| M3.4 | Update [pipeline.py:108](../../../src/attune_rag/pipeline.py:108) docstring to mention the new retrievers as options; no default change. | attune-rag | scaffold | unscoped |
| **M4 — 0.3.0 release** | | | | |
| M4.1 | Add `EmbeddingRetriever`, `HybridRetriever` to `__all__` per [docs/POLICY.md §4](../../POLICY.md#4-add-procedure) (add procedure). | attune-rag | scaffold | unscoped |
| M4.2 | Bump `pyproject.toml` version `0.2.z → 0.3.0`. Classifier stays at `3 - Alpha`. | attune-rag | scaffold | unscoped |
| M4.3 | Roll CHANGELOG `[Unreleased]` into `[0.3.0] — <date>`. New retrievers under `### Added`. `[embeddings]` extra noted. | attune-rag | scaffold | unscoped |
| M4.4 | Update README with a "Semantic retrieval" section pointing at `HybridRetriever` and the `[embeddings]` extra. | attune-rag | scaffold | unscoped |
| M4.5 | Tag + PyPI publish via the existing release workflow (`/attune-release-check` skill + manual `pypi` environment approval). | attune-rag | scaffold | unscoped |

### Dependencies (sketch)

```
M1 (bake-off + skeleton) ─→ M2 (hybrid + α sweep) ─→ M3 (benchmark + paraphrase regression) ─→ M4 (release)
```

M1.4 (model pick) is the critical gate — if no candidate clears the success criterion in [requirements.md §Success criteria](requirements.md#success-criteria-to-be-made-measurable-during-scoping), the spec returns a **negative result** and we defer rather than ship a worse retriever. M2 does not start until M1.4 is decided.

### Definition of done (sketch)

To be finalized during scoping. Initial bullets:

- [ ] 0.3.0 on PyPI with `[embeddings]` extra.
- [ ] `EmbeddingRetriever` and `HybridRetriever` in `__all__` and lock-tested by `tests/unit/test_api_surface.py`.
- [ ] Baseline `queries.yaml` regression unchanged (P@1 ≥ 0.95, R@3 ≥ 1.00, faithfulness ≥ 0.9686) when running `HybridRetriever` at the picked α.
- [ ] Paraphrased `queries_paraphrased.yaml` reports P@1 ≥ 41.25% (11.25% baseline + 30pp absolute lift). Logged in CI, info-only.
- [ ] `pip install attune-rag[embeddings]` total footprint < 150 MB.
- [ ] First-query latency overhead measured and reported. Steady-state per-query overhead < 50 ms.
- [ ] `attune-gui` consumer not affected (no API surface removed; new symbols additive).
- [ ] `docs/specs/embedding-retriever/` includes: this spec, `diagnostic-1.md` (already), `model-bakeoff.md` (M1.4), `alpha-sweep.md` (M2.2), `faithfulness-with-hybrid.md` (M3.3).

### Risks & mitigations (sketch)

See [design.md §Risks & mitigations](design.md#risks--mitigations-sketch). The two highest-impact:

1. **No model clears the success criterion.** M1.4 is the explicit kill-switch; the spec returns a negative result rather than shipping a worse retriever. Diagnostic-1 + bake-off report stand as the artifact even if the build is deferred.
2. **`HybridRetriever` regresses baseline P@1 at any α.** M2.2 sweep + M3.3 faithfulness re-run catch this before M4. The acceptance criterion is "no baseline regression" — if every α regresses, the design needs rework (likely score-normalization shape) before release.

### Out of scope (deferred)

- API-backed embedding providers. Follow-up spec if measured need emerges.
- Vector store integration. In-memory cosine at current corpus size.
- Chunking beyond one-vector-per-entry.
- Default-retriever switch from keyword to hybrid. Separate decision at 0.4.0 or later.
- The `attune-hub` noise-attractor finding from diagnostic-1. Orthogonal task.

### Follow-ups (post-0.3.0)

To be filled during scoping. Expected entries:

- Decide at 0.4.0 whether paraphrased-P@1 becomes a hard CI gate.
- Decide at 0.4.0 whether `HybridRetriever` becomes the docs-recommended default for new users.
- Open `embedding-providers/` spec if there's measured demand for API-backed embeddings.
- Re-author paraphrase set with a second author or from real query logs once usage data exists, to address the authorship-bias caveat in [diagnostic-1.md](diagnostic-1.md#caveats-honest-read).
