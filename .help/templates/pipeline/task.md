---
type: task
name: pipeline-task
feature: pipeline
depth: task
generated_at: 2026-05-16T10:22:03.609395+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Work with the RAG pipeline

Use `RagPipeline` when you need to orchestrate corpus retrieval, prompt assembly, and LLM generation in a single call and get back a structured `RagResult` containing the augmented prompt, citations, confidence score, and elapsed time.

## Prerequisites

- Read access to `src/attune_rag/pipeline.py` and `src/attune_rag/__init__.py`
- A configured corpus that implements `CorpusProtocol`

## Steps

1. **Instantiate `RagPipeline` with your corpus and retriever.**

   ```python
   from attune_rag import RagPipeline, DirectoryCorpus, KeywordRetriever

   corpus = DirectoryCorpus("path/to/docs")
   pipeline = RagPipeline(corpus=corpus, retriever=KeywordRetriever())
   ```

   Pass optional `expander` and `reranker` arguments if you need query expansion or LLM-based reranking.

2. **Run a query.**

   Call `RagPipeline.run()` to retrieve context and assemble a prompt without invoking an LLM:

   ```python
   result = pipeline.run(query="How does caching work?", k=3)
   ```

   Or call `RagPipeline.run_and_generate()` to retrieve context *and* generate a response from an LLM in one step:

   ```python
   answer, result = pipeline.run_and_generate(
       query="How does caching work?",
       provider="openai",
       k=3,
       model="gpt-4o",
   )
   ```

3. **Inspect the `RagResult`.**

   Both methods return a `RagResult` dataclass. Check the fields you need:

   ```python
   print(result.augmented_prompt)   # the assembled prompt
   print(result.citation)           # CitationRecord provenance
   print(result.confidence)         # float confidence score
   print(result.fallback_used)      # True if no grounding context was found
   print(result.elapsed_ms)         # wall-clock time in milliseconds
   print(result.context)            # raw retrieved context
   print(result.claim_citations)    # tuple of ClaimCitation objects
   ```

   If `fallback_used` is `True`, the pipeline found no matching documents and fell back to an ungrounded prompt. Verify your corpus path and query before proceeding.

4. **Choose a prompt variant if needed.**

   Both `run()` and `run_and_generate()` accept a `prompt_variant` argument. The available variants are defined in `PROMPT_VARIANTS`. The default is `"citation"`.

5. **Run the tests to verify your integration.**

   ```
   pytest -k "pipeline"
   ```

## Key files

- `src/attune_rag/pipeline.py` — defines `RagPipeline` and `RagResult`
- `src/attune_rag/__init__.py` — re-exports the public API

## Verify success

Your integration is working correctly when:

- `pipeline.run()` returns a `RagResult` with `fallback_used = False` and a non-empty `context` field
- `result.elapsed_ms` is a positive number
- `pytest -k "pipeline"` passes with no failures
