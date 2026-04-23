---
type: concept
feature: pipeline
depth: concept
generated_at: 2026-04-23T03:32:40.577127+00:00
source_hash: 65f24abb9bb5f4301d29cbd0c7d716a93bfe027e33389ceb15135635b6d7a679
status: generated
---

# Pipeline

## What

The pipeline is a RAG (Retrieval-Augmented Generation) orchestrator that connects your corpus, retrieval system, and LLM in a single call. You provide a query, and it returns both an answer and full provenance showing exactly which sources informed the response.

## How it works

When you call `RagPipeline.run()`, the system follows this flow:

1. **Retrieve** relevant passages from your corpus using the configured retriever
2. **Augment** your original query by injecting the retrieved context into a prompt template
3. **Generate** citations and confidence scores based on source relevance
4. **Package** everything into a `RagResult` with the final prompt, citation details, and timing metadata

The pipeline is LLM-agnostic — it prepares the augmented prompt but doesn't call the language model itself. Use `run_and_generate()` if you want the LLM call included.

## Core components

**RagPipeline** — The orchestrator class that coordinates retrieval and prompt building. Initialize it with a corpus (your knowledge base) and retriever (your search strategy).

**RagResult** — The output containing everything you need: the augmented prompt ready for your LLM, citation records for transparency, confidence scores, and performance timing.

## Fallback behavior

When the retriever finds no relevant context for a query, the pipeline switches to `FALLBACK_PROMPT_TEMPLATE`. This template instructs the LLM to answer honestly about what it knows without inventing project-specific details. The `RagResult.fallback_used` field tells you when this happened.

## Integration points

The pipeline connects to other attune components through two protocols:

- **CorpusProtocol** — Your knowledge base (like `DirectoryCorpus` for file-based sources)
- **RetrieverProtocol** — Your search strategy (like `KeywordRetriever` for text matching)

This design lets you swap corpus types or retrieval algorithms without changing pipeline code.
