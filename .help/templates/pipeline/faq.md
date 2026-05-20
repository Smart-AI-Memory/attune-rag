---
type: faq
name: pipeline-faq
feature: pipeline
depth: faq
generated_at: 2026-05-20T03:20:40.535683+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline FAQ

## What is the pipeline feature?

The pipeline feature provides a lightweight, LLM-agnostic RAG pipeline that wires together a corpus, retriever, prompt assembly, and optional LLM generation. `RagPipeline` is the central class; calling its `run()` method returns a `RagResult` containing the augmented prompt, citation provenance, confidence score, and timing information.

## What's the difference between `run()` and `run_and_generate()`?

`run()` assembles the augmented prompt and retrieval metadata but does not call an LLM — you get back a `RagResult` and can pass the prompt to any model yourself. `run_and_generate()` does the same retrieval and prompt assembly, then calls the LLM provider you specify and returns both the generated text and the `RagResult`.

## What does `RagResult` contain?

`RagResult` is a dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `augmented_prompt` | `str` | The fully assembled prompt sent to the LLM. |
| `citation` | `CitationRecord` | Provenance record for retrieved sources. |
| `confidence` | `float` | Confidence score for the retrieval. |
| `fallback_used` | `bool` | `True` if no grounding context was found and the fallback prompt was used. |
| `elapsed_ms` | `float` | Total pipeline time in milliseconds. |
| `context` | `str` | Retrieved context text (empty string if none). |
| `claim_citations` | `tuple[ClaimCitation, ...]` | Per-claim citation records when using claim-level attribution. |
| `used_native_citations` | `bool` | `True` if the LLM's native citation mechanism was used. |

## What happens when the corpus returns no results?

When no grounding context is found, `RagPipeline` substitutes a fallback prompt that instructs the LLM to answer honestly and avoid fabricating APIs, workflow names, or CLI commands. `RagResult.fallback_used` will be `True` in this case.

## Which prompt variants are available?

The available variants are defined in the `PROMPT_VARIANTS` constant. Pass your chosen variant name as the `prompt_variant` argument to `run()` or `run_and_generate()`. The default is `'citation'`.

## How do I add query expansion or reranking?

Pass a `QueryExpander` instance as `expander` and an `LLMReranker` instance as `reranker` when constructing `RagPipeline`. Both are optional; omitting them gives you straightforward retrieval with no expansion or reranking.

## How do I bring my own corpus or retriever?

`RagPipeline.__init__` accepts any object that satisfies `CorpusProtocol` for `corpus` and `RetrieverProtocol` for `retriever`. Built-in options exported from this module include `DirectoryCorpus`, `AttuneHelpCorpus`, and `KeywordRetriever`.

## How do I debug a pipeline that isn't behaving as expected?

Start by checking `RagResult.fallback_used`, `RagResult.confidence`, and `RagResult.context` — these tell you immediately whether retrieval found anything useful. If retrieval looks correct but generation is wrong, inspect `RagResult.augmented_prompt` to see exactly what was sent to the LLM. To run the pipeline tests, use `pytest -k "pipeline" -v`.

## Where are the source files?

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
