---
type: reference
feature: pipeline
depth: reference
generated_at: 2026-04-23T03:33:06.535660+00:00
source_hash: 65f24abb9bb5f4301d29cbd0c7d716a93bfe027e33389ceb15135635b6d7a679
status: generated
---

# Pipeline reference

Build RAG workflows that combine corpus retrieval, prompt augmentation, and citation tracking.

## Classes

| Class | Description |
|-------|-------------|
| `RagPipeline` | Orchestrate retrieval-augmented generation workflows |
| `RagResult` | Structured output from pipeline execution |

### RagPipeline

Coordinates corpus search, context injection, and provenance tracking for LLM-agnostic RAG workflows.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `corpus: CorpusProtocol \| None = None, retriever: RetrieverProtocol \| None = None` | `None` | Initialize pipeline with corpus and retriever components |
| `run` | `query: str, k: int = 3, prompt_variant: str = 'citation'` | `RagResult` | Execute retrieval and prompt augmentation workflow |
| `run_and_generate` | `query: str, provider: LLMProvider \| str, k: int = 3, model: str \| None = None, max_tokens: int = 2048, prompt_variant: str = 'citation'` | `tuple[str, RagResult]` | Execute pipeline and generate LLM response |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `corpus` | `CorpusProtocol` | Access the underlying corpus instance |

### RagResult

Structured output capturing retrieval context, citations, performance metrics, and fallback status.

#### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `augmented_prompt` | `str` |  | Prompt text with injected retrieval context |
| `citation` | `CitationRecord` |  | Source attribution and provenance data |
| `confidence` | `float` |  | Retrieval relevance score |
| `fallback_used` | `bool` |  | Whether fallback prompt was used when no context found |
| `elapsed_ms` | `float` |  | Pipeline execution time in milliseconds |
| `context` | `str` | `''` | Retrieved text snippets used for augmentation |

## Constants

| Constant | Description |
|----------|-------------|
| `FALLBACK_PROMPT_TEMPLATE` | Template used when no grounding context is found in the corpus |
| `__version__` | Package version string |
| `__all__` | Public API exports |

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

## Tags

`pipeline`, `orchestration`, `rag`, `result`
