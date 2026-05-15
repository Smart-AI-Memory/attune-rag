---
type: task
name: pipeline-task
feature: pipeline
depth: task
generated_at: 2026-05-15T20:01:28.744228+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Work with the RAG pipeline

Use `RagPipeline` when you want to run retrieval, prompt assembly, and LLM generation in a single call and receive a structured `RagResult` containing the answer, confidence score, and citation provenance.

## Prerequisites

- Read access to `src/attune_rag/pipeline.py` and `src/attune_rag/__init__.py`
- A configured corpus that implements `CorpusProtocol`

## Run a basic query

1. **Import `RagPipeline` and instantiate it** with your corpus:

   ```python
   from attune_rag import RagPipeline, DirectoryCorpus

   corpus = DirectoryCorpus("path/to/docs")
   pipeline = RagPipeline(corpus=corpus)
   ```

2. **Call `pipeline.run()`** with your query:

   ```python
   result = pipeline.run(query="What is the retention policy?", k=3)
   ```

   `run()` returns a `RagResult`. The fields you'll use most often are:

   | Field | Type | Contains |
   |---|---|---|
   | `augmented_prompt` | `str` | The prompt sent to the LLM |
   | `citation` | `CitationRecord` | Source provenance for the answer |
   | `confidence` | `float` | Retrieval confidence score |
   | `fallback_used` | `bool` | `True` if no grounding context was found |
   | `context` | `str` | Retrieved context passages |
   | `elapsed_ms` | `float` | Total wall-clock time for the call |

3. **Check `fallback_used`** before trusting the answer:

   ```python
   if result.fallback_used:
       print("No grounding context found — answer may be speculative.")
   else:
       print(result.augmented_prompt)
       print(result.citation)
   ```

## Run retrieval and generation together

If you want the pipeline to call an LLM and return the generated text alongside the `RagResult`, use `run_and_generate()` instead:

```python
text, result = pipeline.run_and_generate(
    query="Summarise the onboarding process",
    provider="openai",
    k=3,
    model="gpt-4o",
    max_tokens=512,
    prompt_variant="citation",
)
print(text)
print(result.claim_citations)
```

Pass `use_native_citations=True` to let the LLM provider handle citation formatting natively; `result.used_native_citations` will reflect whether that path was taken.

## Extend the pipeline

1. **Add an expander or reranker** by passing them to the constructor:

   ```python
   from attune_rag import QueryExpander, LLMReranker

   pipeline = RagPipeline(
       corpus=corpus,
       expander=QueryExpander(),
       reranker=LLMReranker(),
   )
   ```

   `RagPipeline.__init__` accepts `corpus`, `retriever`, `expander`, and `reranker` — all optional.

2. **Swap the retriever** if the default `KeywordRetriever` does not suit your corpus. Pass any object that implements `RetrieverProtocol`:

   ```python
   pipeline = RagPipeline(corpus=corpus, retriever=my_custom_retriever)
   ```

3. **Run the pipeline tests** to verify your changes:

   ```bash
   pytest -k "pipeline"
   ```

## Verify success

After calling `pipeline.run()`, confirm all of the following:

- `result.augmented_prompt` is a non-empty string
- `result.confidence` is greater than `0.0`
- `result.fallback_used` is `False` (for a query your corpus covers)
- `result.elapsed_ms` is a positive number

If `result.fallback_used` is `True` for a query you expect the corpus to answer, check that your `DirectoryCorpus` path points to the correct documents and that `k` is large enough to retrieve relevant passages.

## Key files

| File | Purpose |
|---|---|
| `src/attune_rag/pipeline.py` | `RagPipeline` and `RagResult` definitions |
| `src/attune_rag/__init__.py` | Public exports including all pipeline symbols |
