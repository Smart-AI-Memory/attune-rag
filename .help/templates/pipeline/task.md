---
type: task
name: pipeline-task
feature: pipeline
depth: task
generated_at: 2026-05-20T02:44:13.905763+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Work with the RAG pipeline

Use `RagPipeline` when you want to combine corpus retrieval, prompt assembly, and LLM generation into a single call that returns a `RagResult` containing the augmented prompt, citations, confidence score, and elapsed time.

## Prerequisites

- Access to the project source code
- A corpus and retriever available to pass to `RagPipeline.__init__`

## Steps

1. **Instantiate `RagPipeline`** with your corpus and retriever:

   ```python
   from attune_rag import RagPipeline, DirectoryCorpus, KeywordRetriever

   corpus = DirectoryCorpus("docs/")
   retriever = KeywordRetriever()
   pipeline = RagPipeline(corpus=corpus, retriever=retriever)
   ```

   Optionally pass a `QueryExpander` or `LLMReranker` to improve retrieval quality.

2. **Run the pipeline** by calling `pipeline.run()` with your query:

   ```python
   result = pipeline.run(query="How does citation work?", k=3, prompt_variant="citation")
   ```

   This returns a `RagResult`. Check `result.fallback_used` — if `True`, the corpus contained no matching context and the pipeline fell back to an unconditioned prompt.

3. **Inspect the result fields** you need:

   | Field | Type | What it contains |
   |---|---|---|
   | `augmented_prompt` | `str` | The fully assembled prompt sent to the LLM |
   | `citation` | `CitationRecord` | Provenance for the retrieved context |
   | `confidence` | `float` | Retrieval confidence score |
   | `fallback_used` | `bool` | Whether the fallback prompt was used |
   | `elapsed_ms` | `float` | Total pipeline wall-clock time in milliseconds |
   | `context` | `str` | Raw retrieved context |
   | `claim_citations` | `tuple[ClaimCitation, ...]` | Per-claim citation records |
   | `used_native_citations` | `bool` | Whether the LLM's native citation mechanism was used |

4. **Generate a response** in one step by calling `run_and_generate()` if you want the pipeline to call the LLM directly:

   ```python
   answer, result = pipeline.run_and_generate(
       query="How does citation work?",
       provider="openai",
       k=3,
       model="gpt-4o",
       max_tokens=2048,
       prompt_variant="citation",
   )
   ```

   `answer` is the LLM's response string. `result` is the same `RagResult` you would get from `run()`.

5. **Run the pipeline tests** to confirm your integration works:

   ```
   pytest -k "pipeline"
   ```

## Key files

- `src/attune_rag/pipeline.py` — `RagPipeline` and `RagResult` definitions
- `src/attune_rag/__init__.py` — public exports

## Verify success

After calling `pipeline.run()` or `pipeline.run_and_generate()`, confirm that:

- `result.augmented_prompt` is a non-empty string containing your query
- `result.fallback_used` is `False` (meaning the corpus returned relevant context)
- `result.elapsed_ms` is a positive number
- `pytest -k "pipeline"` exits with no failures
