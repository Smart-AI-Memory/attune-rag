---
type: comparison
name: pipeline-comparison
feature: pipeline
depth: comparison
generated_at: 2026-05-20T03:20:40.545203+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: verified
---

# Comparison: Pipeline orchestration approaches

## Context

`RagPipeline` wires together four concerns — corpus retrieval, optional query expansion and reranking, prompt assembly, and LLM generation — into two callable entry points:

- **`run()`** — returns a `RagResult` with a fully assembled prompt and `CitationRecord` provenance, but does **not** call the answer provider. You supply the LLM call yourself.
- **`run_and_generate()`** — returns an answer and a `RagResult`. Strict zero-hit queries return a local refusal. Other queries use an answer provider.

The sections below compare these two approaches and describe the narrower alternatives they replace.

---

## Feature comparison

| Capability | `run()` | `run_and_generate()` |
|---|---|---|
| Retrieves top-*k* documents from corpus | ✓ | ✓ |
| Expands query via `QueryExpander` | ✓ (if configured) | ✓ (if configured) |
| Reranks hits via `LLMReranker` | ✓ (if configured) | ✓ (if configured) |
| Assembles augmented prompt | ✓ | ✓ |
| Returns `CitationRecord` provenance | ✓ | ✓ |
| Returns per-claim citations (`claim_citations`) | ✗ | With native citations |
| Records elapsed time (`elapsed_ms`) | ✓ | ✓ |
| Signals fallback when no grounding found (`fallback_used`) | ✓ | ✓ |
| Calls an LLM and returns generated text | ✗ | ✓ |
| Supports native citation mode (`use_native_citations`) | ✗ | ✓ |
| Lets you choose your own LLM call site | ✓ | ✗ |
| Requires an `LLMProvider` | ✗ | ✓ |

---

## Fallback behaviour

With zero retrieval hits, `run()` and `run_and_generate()` set `fallback_used=True`. With `prompt_variant="strict"`, they also set `abstained=True`, and `run_and_generate()` returns `The provided context does not cover this question.` locally. Other variants use the plain fallback prompt.

---

## When NOT to use `RagPipeline` directly

- **Single-step retrieval without prompt assembly.** If you only need ranked document hits, use `KeywordRetriever` (or a custom `RetrieverProtocol`) directly. `RagPipeline` adds prompt-building overhead you won't use.
- **Exploratory or one-off queries.** A short script that calls a retriever and formats its own prompt is simpler to reason about than instantiating a full pipeline for a single run.
- **Custom prompt logic that diverges from `PROMPT_VARIANTS`.** If the built-in prompt variants don't fit your use case, building the augmented prompt yourself with `build_augmented_prompt` gives you more control without patching pipeline internals.
- **Multi-pipeline orchestration.** If your application chains multiple retrieval passes or fans out across corpora, coordinate that logic in a layer above `RagPipeline` rather than nesting pipeline calls.

---

## Use X when…

**Use `run()`** when you want full RAG orchestration — retrieval, reranking, prompt assembly, and provenance — but need to control the LLM call yourself (to apply streaming, token budgets, retry logic, or a provider not yet supported by `LLMProvider`).

**Use `run_and_generate()`** when you want a single call that handles everything end-to-end and you are happy to delegate the LLM invocation to the pipeline. This is the right default for most production use cases where the built-in `LLMProvider` covers your target model.

**Use a retriever directly** (e.g. `KeywordRetriever`) when you don't need prompt assembly or citation tracking — for example, a search-results UI that displays raw document excerpts.

---

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
