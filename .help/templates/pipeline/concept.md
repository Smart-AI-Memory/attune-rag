---
type: concept
name: pipeline-concept
feature: pipeline
depth: concept
generated_at: 2026-05-20T03:20:40.511210+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline

## Overview

`RagPipeline` is a lightweight, LLM-agnostic orchestrator that connects a document corpus, a retriever, and a prompt builder so that a single `run()` call returns a grounded, citable answer.

## How the pieces fit together

When you call `RagPipeline.run(query, k=3)`, the pipeline:

1. Retrieves the top-`k` passages from the attached `CorpusProtocol` using the configured `RetrieverProtocol`.
2. Optionally expands the query with `QueryExpander` and re-ranks hits with `LLMReranker`.
3. Assembles an augmented prompt from the retrieved context using one of the built-in `PROMPT_VARIANTS` (default: `"citation"`).
4. Returns a `RagResult` that packages the augmented prompt, a `CitationRecord`, a confidence score, elapsed time, and a `fallback_used` flag.

If no grounding context is found, the pipeline substitutes a fallback prompt that instructs the LLM to answer honestly rather than invent APIs, workflow names, or CLI commands.

To go further in one step, `run_and_generate()` accepts an `LLMProvider` and returns both the generated text and the `RagResult`, making it straightforward to hand the result directly to a UI or downstream service.

## Core data structures

### `RagPipeline`

The central orchestrator. Constructed with optional components:

```python
RagPipeline(
    corpus=...,      # CorpusProtocol — document store to retrieve from
    retriever=...,   # RetrieverProtocol — scoring and ranking logic
    expander=...,    # QueryExpander — optional query rewriting
    reranker=...,    # LLMReranker — optional LLM-based re-ranking
)
```

You can read back the attached corpus at any time via the `corpus` property.

### `RagResult`

The structured output of every `run()` call. Key fields:

| Field | Type | What it tells you |
|---|---|---|
| `augmented_prompt` | `str` | The prompt actually sent to the LLM, with retrieved context injected |
| `citation` | `CitationRecord` | Source provenance for the answer |
| `claim_citations` | `tuple[ClaimCitation, ...]` | Per-claim attribution when native citations are active |
| `confidence` | `float` | Retrieval confidence score |
| `fallback_used` | `bool` | `True` when no corpus context was found |
| `elapsed_ms` | `float` | End-to-end pipeline latency |
| `context` | `str` | Raw retrieved context passed to the prompt |

## When this matters

Use `RagPipeline` when you need retrieval-augmented generation that is not coupled to a specific LLM provider. Because the pipeline separates corpus management, retrieval, and prompt assembly into swappable components, you can change any one of them — for example, substituting `KeywordRetriever` for a dense retriever — without touching the rest of your application.
