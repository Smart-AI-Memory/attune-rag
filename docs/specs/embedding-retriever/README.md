# Spec: embedding-retriever

> **Status: deferred (permanent for the attune-help corpus context as of 2026-05-21) — the [alias-expansion-sweep](../alias-expansion-sweep/) closed the paraphrase gap (R@3 28.75% → 100% after D4, baseline still 100%/100%) without any new dependency. Revival for the bundled corpus would require evidence the alias mechanism can't close a gap that matters for shipped usage; none observed. For arbitrary user corpora the defer is *scope-specific, not absolute* — see "Scope of the defer" below.**

- **Owner:** Patrick
- **Created:** 2026-05-21
- **Deferred:** 2026-05-21 (same day, after D2 and D3 results landed)
- **Defer made permanent:** 2026-05-21 (after [alias-expansion-sweep](../alias-expansion-sweep/) M13.1 confirmed paraphrased R@3 = 96.25%, well above the M13.2 ≥70% revival threshold)
- **Target version (if revived):** 0.3.0+ (post-0.2.0 cut, additive only)
- **Predecessor work:** [diagnostic-1.md](diagnostic-1.md), [diagnostic-2.md](diagnostic-2.md), [diagnostic-3.md](diagnostic-3.md).
- **Entry condition (D1):** Δ P@1 > 15pp on the paraphrase set → STRONG verdict. **Met:** observed Δ P@1 = −86.25pp.
- **Defer condition (D2 + D3):** dependency-free alternatives close ≥ 30pp of the paraphrased R@3 gap with smaller or zero baseline cost. **Met:** D3 +50pp R@3 on bug-predict cluster with zero regression, zero dep; D2 +51pp R@3 overall (but with −10pp baseline P@1 cost, ruling it out as a default but viable as opt-in).
- **Permanent-defer condition (M13.2):** alias-expansion-sweep lands paraphrased R@3 ≥ 70% with no baseline regression. **Met at 96.25% / 100% baseline at M13.2; closed to 100% / 100% at [D4](diagnostic-4.md).**

## Scope of the defer

The permanent defer is **scope-specific, not absolute**. Added 2026-05-21 alongside the [v1.0.0 Phase 5 scope decision](../v1.0.0-release/design.md#phase-5-scope-decided-2026-05-21), which committed v1.0.0 to the framework framing.

- **Permanent for the attune-help corpus context.** The alias-expansion sweep closed paraphrased R@3 to 100% on the 80-query regression set. For users consuming the bundled `AttuneHelpCorpus`, the embedding-retriever question is settled — `KeywordRetriever` + the override mechanism is the answer, with no embedding dependency required.
- **Viable for arbitrary user corpora** (post-v1.0.0 framework framing). The [`user-corpus-onboarding`](../user-corpus-onboarding/) spec (Phase 5 of the v1.0 roadmap; scaffolded Phase 4 W2) ships a quality harness so users can measure retrieval quality on their own markdown corpora. **If the harness consistently surfaces gaps that the frontmatter-alias + override path can't close for a user-corpus class**, the embedding-retriever revival case re-opens — for that corpus class, not for the bundled one.
- **Revival path preserved.** The D1–D4 diagnostic artifacts remain on disk specifically so a future revival can pick up the evaluation framework without re-deriving it. The 80-query paraphrase set is the harness shape; future user-corpus paraphrase sets would slot into the same shape. The diagnostic scripts (`run_diagnostic_1.py` through `run_diagnostic_3.py`) are reusable drivers, not attune-help-specific.

The defer is "permanent" in the sense that it closes the *original question* (does attune-help's retrieval need embeddings?). It does not foreclose embeddings as a future feature for the broader framework framing — a future revival would scope a new spec (e.g. `docs/specs/embedding-retriever-for-user-corpora/`) using this one as the methodology parent.

## Purpose (deferred — kept for archival context)

Add a semantic retrieval path to `attune-rag` so the pipeline can surface the correct template when the user's query shares no tokens with the target's path / summary / aliases. Diagnostic-1 established that the current `KeywordRetriever` collapses on paraphrased queries (P@1 97.5% → 11.25%, R@3 100% → 28.75%), so this was framed as a measured-need addition, not a speculative one.

**Why deferred (2026-05-21):** D2 and D3 found that the existing `QueryExpander` (+51pp R@3 on paraphrased) and zero-dep alias expansion (+50pp R@3 on the bug-predict cluster) close most of the gap without adding a new dependency. At current corpus scale, the embedding-retriever case no longer holds against these alternatives. The artifacts (paraphrase set, diagnostic scripts, summaries) stay on disk as the supporting evidence bundle; the spec itself does not promote out of `deferred` unless an alias-expansion sweep across all under-served clusters leaves a meaningful residual.

The default retriever stays `KeywordRetriever`. No new retriever ships from this spec.

## What this spec is

| Layer | Today | This spec |
|---|---|---|
| Default retriever | `KeywordRetriever` | unchanged |
| Public `RetrieverProtocol` | `retrieve(query, corpus, k) -> Iterable[RetrievalHit]` | unchanged |
| Embedding-backed retriever | does not exist | **added as `EmbeddingRetriever`** |
| Hybrid combiner | does not exist | **added as `HybridRetriever`** (keyword + embedding blend) |
| Embedding cache on `RetrievalEntry` | tokens only (`_tokens_cache`) | mirrors with `_embeddings_cache` keyed by model name |
| Install footprint | no embedding deps | **opt-in `[embeddings]` extra** (small local model) |
| Benchmark gate | baseline `queries.yaml` only | **paraphrase set promoted to permanent regression** (gate decision deferred to scoping) |
| API-side embedding providers | none | **deferred to a follow-up spec** |

## What this spec is NOT

- **Not a `KeywordRetriever` replacement.** Keyword still wins on exact feature-name queries and is the fast/free default. The hybrid combiner is the recommended user-facing path; pure embedding stays available as a building block.
- **Not an OpenAI / Anthropic embeddings adapter.** Local model only in this scope. API-backed embeddings can come later if there's a measured need.
- **Not a chunking redesign.** Entries are short (path + summary + content preview ≤ 500 chars); embed each entry as a single vector. Multi-vector / passage-level chunking is a follow-up if the paraphrase benchmark plateaus.
- **Not a faithfulness change.** Diagnostic-1 measured retrieval only. Whether better retrieval moves end-to-end faithfulness is a separate diagnostic.
- **Not coupled to the 0.2.0 cut.** This is additive surface and targets 0.3.0; it does not affect the SemVer freeze in [api-v0.2.0-cut/](../api-v0.2.0-cut/).

## Spec files

- [`requirements.md`](requirements.md) — entry gates (diagnostic passed), functional requirements, success criteria, open questions for scoping.
- [`design.md`](design.md) — model choice, hybrid algorithm sketch, cache shape, `[embeddings]` extra, risks.
- [`tasks.md`](tasks.md) — four milestones at the **scaffold** level (M1 model + index, M2 hybrid combiner, M3 benchmark integration, M4 release). Unscoped until the `/spec` pass.
- [`diagnostic-1.md`](diagnostic-1.md) — entry-condition artifact. Read this first.
- [`run_diagnostic_1.py`](run_diagnostic_1.py) — reusable driver for the same paraphrase probe against any future retriever.
- [`diagnostic-1.run-output.md`](diagnostic-1.run-output.md) — captured run output for reproducibility.

## Activation path

This spec promotes from "scoping" to "approved" when the `/spec` scoping pass:

1. Picks the local embedding model (provisional: `sentence-transformers/all-MiniLM-L6-v2`, but `fastembed` ONNX equivalents are in scope to evaluate).
2. Decides the hybrid blend algorithm (provisional: convex combination `α · norm(keyword) + (1−α) · cosine`, with α as a class attribute sweepable by the benchmark).
3. Decides whether paraphrased-P@1 gates CI or stays an info-only signal at first (recommended: info-only for one minor, then gate).
4. Fills in `tasks.md` with concrete acceptance criteria and dependency arrows.

Calendar target: opens for `/spec` scoping after the 0.2.0 cut lands (i.e., post-[api-v0.2.0-cut/](../api-v0.2.0-cut/) M2). Earlier is acceptable if the diagnostic findings shift priority.

## Out of scope (explicit)

- OpenAI / Anthropic embedding providers (follow-up spec if measured need emerges).
- Vector store integration (FAISS, qdrant, pgvector). Corpus size is hundreds of docs; in-memory cosine is sufficient and avoids the dep.
- Chunking strategy beyond "one vector per entry."
- Replacing `KeywordRetriever` as the pipeline default.
- Touching the existing `[claude]` / `[gemini]` extras or LLM-side adapters.
- Re-litigating the `concepts/` vs `tasks/` ranking decision (see [`retrieval_bug_code_quality.md`](../../../.claude/projects/-Users-patrickroebuck-attune-rag/memory/retrieval_bug_code_quality.md) for history).
- The `attune-hub` noise-attractor finding from diagnostic-1 (orthogonal; tracked as a separate task).
