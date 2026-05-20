---
type: concept
name: pipeline-concept
feature: pipeline
depth: concept
generated_at: 2026-05-20T02:44:13.899731+00:00
source_hash: f5cc845ee3957a76674328c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline

## What the pipeline is

`RagPipeline` is the central orchestrator in attune-rag: it connects a corpus, a retriever, and an LLM into a single call that returns a grounded, cited answer.

When you call `RagPipeline.run()`, the pipeline retrieves the most relevant documents from the corpus, assembles an augmented prompt from those documents and your query, and returns a `RagResult` that bundles the prompt, a `CitationRecord`, a confidence score, and timing information. If no grounding context is found, the pipeline falls back to an honest, instruction-constrained prompt rather than fabricating an answer.

## Core responsibilities

`RagPipeline` coordinates four concerns that would otherwise be wired together manually:

| Concern | How the pipeline handles it |
|---|---|
| **Document retrieval** | Delegates to a `RetrieverProtocol` implementation; defaults to `KeywordRetriever` if none is supplied |
| **Query expansion** | Optionally passes the query through a `QueryExpander` before retrieval |
| **Reranking** | Optionally reorders retrieved hits with an `LLMReranker` |
| **Prompt assembly** | Calls `build_augmented_prompt` with a named variant (e.g., `'citation'`) to format retrieved context into the final prompt |

This separation means you can swap any layer — the corpus, the retriever, the reranker — without touching the others.

## The two central types

**`RagResult`** is the value `RagPipeline.run()` returns. Every field answers a specific question a caller might ask after the run:

| Field | What it tells you |
|---|---|
| `augmented_prompt` | The fully assembled prompt that was (or would be) sent to the LLM |
| `citation` | A `CitationRecord` identifying which sources grounded the answer |
| `confidence` | How strongly the retrieved context supports the answer |
| `fallback_used` | Whether retrieval found anything (`True` means the answer is ungrounded) |
| `elapsed_ms` | Wall-clock time for the full pipeline run |
| `context` | The raw retrieved text that was injected into the prompt |
| `claim_citations` | Per-claim `ClaimCitation` tuples when native citation mode is active |
| `used_native_citations` | Whether the LLM's own citation mechanism was used instead of post-hoc attribution |

**`RagPipeline`** is constructed once and reused across queries. Its `corpus` property exposes the attached `CorpusProtocol` so callers can inspect or replace the document store independently of the pipeline itself.

## Two ways to run the pipeline

`RagPipeline` exposes two entry points depending on whether you supply your own LLM:

- **`run(query, k, prompt_variant)`** — Retrieves context and assembles the augmented prompt, but stops before generation. Use this when you want to call the LLM yourself or inspect the prompt before sending it.
- **`run_and_generate(query, provider, ...)`** — Does everything `run()` does, then calls the specified `LLMProvider` and returns both the generated text and the full `RagResult`. Use this for end-to-end answers in a single call.

The `prompt_variant` parameter (default `'citation'`) selects a named template from `PROMPT_VARIANTS`, giving you control over how retrieved context is framed without rewriting prompt logic.

## When `fallback_used` is `True`

If retrieval finds no relevant documents, the pipeline substitutes `FALLBACK_PROMPT_TEMPLATE` instead of an augmented prompt. That template instructs the LLM to answer honestly about the limits of its knowledge and explicitly forbids inventing attune APIs, workflow names, or CLI commands. The `RagResult` still comes back with `fallback_used=True` so callers can decide whether to surface a warning or retry with a broader query.
