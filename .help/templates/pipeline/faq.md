---
type: faq
name: pipeline-faq
feature: pipeline
depth: faq
generated_at: 2026-05-20T03:20:40.535683+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: verified
---

# Pipeline FAQ

## What is the pipeline feature?

The pipeline feature provides a lightweight, LLM-agnostic RAG pipeline that wires together a corpus, retriever, prompt assembly, and optional LLM generation. `RagPipeline` is the central class; calling its `run()` method returns a `RagResult` containing the augmented prompt, citation provenance, confidence score, and timing information.

## What's the difference between `run()` and `run_and_generate()`?

`run()` assembles the augmented prompt and retrieval metadata without calling the answer provider — you get back a `RagResult` and can pass the prompt to any model yourself. `run_and_generate()` does the same retrieval and prompt assembly, then returns an answer and its `RagResult`. Strict zero-hit queries return a local refusal.

## What does `RagResult` contain?

`RagResult` is a dataclass with these fields:

| Field | Type | Description |
|---|---|---|
| `augmented_prompt` | `str` | Assembled legacy prompt. Empty for native generation. |
| `citation` | `CitationRecord` | Provenance record for retrieved sources. |
| `confidence` | `float` | Confidence score for the retrieval. |
| `fallback_used` | `bool` | `True` when retrieval returned zero hits. |
| `elapsed_ms` | `float` | Total pipeline time in milliseconds. |
| `context` | `str` | Retrieved context text (empty string if none). |
| `claim_citations` | `tuple[ClaimCitation, ...]` | Per-claim citation records when using claim-level attribution. |
| `used_native_citations` | `bool` | `True` if the LLM's native citation mechanism was used. |
| `abstained` | `bool` | The pipeline enforced a strict zero-hit refusal. |
| `generation_documents` | `tuple[CitationDocument, ...]` | Ordered titles and texts supplied for native generation. |

## What happens when the corpus returns no results?

With zero retrieval hits, `run()` and `run_and_generate()` set `fallback_used=True`. With `prompt_variant="strict"`, they also set `abstained=True`, and `run_and_generate()` returns `The provided context does not cover this question.` locally. Other variants use the plain fallback prompt.

## Which prompt variants are available?

The available variants are defined in the `PROMPT_VARIANTS` constant. Pass your chosen variant name as the `prompt_variant` argument to `run()` or `run_and_generate()`. The default is `'citation'`.

## How do I add query expansion or reranking?

Pass a `QueryExpander` instance as `expander` and an `LLMReranker` instance as `reranker` when constructing `RagPipeline`. Both are optional; omitting them gives you straightforward retrieval with no expansion or reranking.

## How do I bring my own corpus or retriever?

`RagPipeline.__init__` accepts any object that satisfies `CorpusProtocol` for `corpus` and `RetrieverProtocol` for `retriever`. Built-in options exported from this module include `DirectoryCorpus`, `AttuneHelpCorpus`, and `KeywordRetriever`.

## How do I debug a pipeline that isn't behaving as expected?

Start by checking `RagResult.fallback_used`, `RagResult.confidence`, and `RagResult.context` — these tell you immediately whether retrieval found anything useful. If retrieval looks correct but generation is wrong, inspect `augmented_prompt` for legacy generation or `generation_documents` for the ordered native evidence. To run the pipeline tests, use `pytest -k "pipeline" -v`.

## Where are the source files?

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
