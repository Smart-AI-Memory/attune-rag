---
type: note
name: pipeline-note
feature: pipeline
depth: note
generated_at: 2026-05-20T03:20:40.542730+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Note: pipeline

## Context

The `pipeline` module (`src/attune_rag/pipeline.py`) is a lightweight, LLM-agnostic RAG pipeline. It coordinates four optional components — a corpus, a retriever, a query expander, and a reranker — to retrieve grounding context, assemble a prompt, and optionally call an LLM, all in a single method call.

## Content

Two public types form the core of the module:

- **`RagPipeline`** — accepts a `CorpusProtocol`, a `RetrieverProtocol`, an optional `QueryExpander`, and an optional `LLMReranker` at construction time. Its two main methods are:
  - `run(query, k, prompt_variant)` — retrieves up to `k` documents, assembles an augmented prompt, and returns a `RagResult` without calling an LLM.
  - `run_and_generate(query, provider, ...)` — does the same, then calls the specified `LLMProvider` and returns `(answer_str, RagResult)`.

- **`RagResult`** — the dataclass returned by both methods. Key fields include `augmented_prompt`, `citation` (`CitationRecord`), `confidence`, `fallback_used`, `elapsed_ms`, `context`, `claim_citations`, and `used_native_citations`. When the retriever finds no grounding context, `fallback_used` is `True` and the pipeline substitutes `FALLBACK_PROMPT_TEMPLATE`, which instructs the LLM not to invent APIs, workflow names, or CLI commands.

The `prompt_variant` parameter selects from the named templates in `PROMPT_VARIANTS`. The default variant is `'citation'`.

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
