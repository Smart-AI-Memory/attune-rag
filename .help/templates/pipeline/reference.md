---
type: reference
name: pipeline-reference
feature: pipeline
depth: reference
generated_at: 2026-05-20T02:44:13.910492+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline reference

Use `RagPipeline` to orchestrate retrieval, prompt assembly, and LLM generation in a single call. Each run returns a `RagResult` containing the augmented prompt, `CitationRecord` provenance, and confidence metadata.

## Classes

| Class | Description |
|-------|-------------|
| `RagResult` | Dataclass returned by `RagPipeline.run`. Holds the augmented prompt, citation record, confidence score, and timing data. |
| `RagPipeline` | LLM-agnostic RAG pipeline that composes a corpus, retriever, optional query expander, and optional reranker. |

### RagResult

`RagResult` is a dataclass. The fields below are set by `RagPipeline.run` and available on every result object.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `augmented_prompt` | `str` | — |
| `citation` | `CitationRecord` | — |
| `confidence` | `float` | — |
| `fallback_used` | `bool` | — |
| `elapsed_ms` | `float` | — |
| `context` | `str` | `''` |
| `claim_citations` | `tuple[ClaimCitation, ...]` | `()` |
| `used_native_citations` | `bool` | `False` |

### RagPipeline

#### Constructor

```python
RagPipeline(
    corpus: CorpusProtocol | None = None,
    retriever: RetrieverProtocol | None = None,
    expander: QueryExpander | None = None,
    reranker: LLMReranker | None = None,
) -> None
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `corpus` | `CorpusProtocol` | The corpus this pipeline retrieves from. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `run` | `query: str`, `k: int = 3`, `prompt_variant: str = 'citation'` | `RagResult` | Retrieves context, assembles the augmented prompt, and returns a `RagResult`. |
| `run_and_generate` | `query: str`, `provider: LLMProvider \| str`, `k: int = 3`, `model: str \| None = None`, `max_tokens: int = 2048`, `prompt_variant: str = 'citation'`, `use_native_citations: bool = False` | `tuple[str, RagResult]` | Runs retrieval and prompt assembly, then calls the specified LLM provider and returns the generated text alongside the `RagResult`. |

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `__version__` | `str` | `'0.1.19'` |
| `FALLBACK_PROMPT_TEMPLATE` | `str` | `"### USER REQUEST\n\n{query}\n\n### INSTRUCTION\n\nNo grounding context was found in the corpus for this\nrequest. Answer honestly about what you do and do not\nknow. Do not invent attune APIs, workflow names, or CLI\ncommands. If the user is asking about something outside\nthe corpus's scope, say so."` |
| `_CACHE_SPLIT` | `str` | `'\n### USER REQUEST\n'` |

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

## Tags

`pipeline`, `orchestration`, `rag`, `result`
