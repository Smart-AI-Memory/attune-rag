---
type: concept
name: pipeline-concept
feature: pipeline
depth: concept
generated_at: 2026-05-16T10:22:03.603342+00:00
source_hash: f5cc845ee3957a76674362a8c142ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline

## What the pipeline does

`RagPipeline` is a lightweight, LLM-agnostic orchestrator that takes a query, retrieves relevant documents from a corpus, assembles a grounded prompt, and optionally calls an LLM — all in a single method call.

When you call `RagPipeline.run(query)`, the pipeline:

1. Retrieves the top-k documents from the attached corpus using the configured retriever
2. Optionally expands the query (via `QueryExpander`) and reranks results (via `LLMReranker`)
3. Builds an augmented prompt from the retrieved context using one of the available `PROMPT_VARIANTS`
4. Returns a `RagResult` containing the prompt, citation provenance, confidence score, and timing

If no grounding context is found, the pipeline sets `fallback_used = True` on the result and substitutes a `FALLBACK_PROMPT_TEMPLATE` that instructs the LLM to answer honestly rather than invent details.

## Core components

| Component | Role |
|-----------|------|
| `RagPipeline` | The orchestrator. Holds references to the corpus, retriever, expander, and reranker, and coordinates them on each `run()` call. |
| `RagResult` | The output record. Carries `augmented_prompt`, `citation`, `confidence`, `fallback_used`, `elapsed_ms`, `context`, and optionally `claim_citations`. |
| `CorpusProtocol` | The document store the pipeline retrieves from. Accessible via `RagPipeline.corpus`. |
| `RetrieverProtocol` | Selects the top-k documents from the corpus. `KeywordRetriever` is the built-in implementation. |
| `QueryExpander` | Optional. Broadens the query before retrieval to improve recall. |
| `LLMReranker` | Optional. Re-scores retrieved documents using an LLM before prompt assembly. |

## Two ways to run

**`run(query, k, prompt_variant)`** — retrieval and prompt assembly only. Returns a `RagResult` with `augmented_prompt` ready to pass to any LLM you choose. Use this when you control the generation step yourself.

**`run_and_generate(query, provider, ...)`** — does everything `run()` does, then calls the specified `LLMProvider` and returns `(response_text, RagResult)`. Use this when you want the pipeline to handle generation as well.

## How the pieces fit together

```
query
  │
  ▼
QueryExpander (optional)
  │  expanded query
  ▼
RetrieverProtocol  ◄──── CorpusProtocol
  │  top-k RetrievalHits
  ▼
LLMReranker (optional)
  │  reranked hits
  ▼
build_augmented_prompt  ◄──── PROMPT_VARIANTS
  │
  ▼
RagResult
  ├── augmented_prompt
  ├── citation / claim_citations
  ├── confidence
  ├── fallback_used
  └── elapsed_ms
```

Each stage is pluggable: swap in a different corpus, retriever, or reranker by passing it to `RagPipeline.__init__`. The pipeline itself stays the same.

## When the fallback fires

If the retriever returns no usable context, `RagPipeline` substitutes `FALLBACK_PROMPT_TEMPLATE` instead of an augmented prompt and marks `RagResult.fallback_used = True`. The template explicitly instructs the LLM not to invent API names, workflow names, or CLI commands — making the absence of grounding context visible and safe by default.
