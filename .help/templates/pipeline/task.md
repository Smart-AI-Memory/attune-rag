---
type: task
name: pipeline-task
feature: pipeline
depth: task
generated_at: 2026-05-20T03:20:40.516916+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Run a RAG pipeline

Use `RagPipeline` when you want to retrieve grounded context from a corpus, assemble a prompt, and generate a cited answer — all in a single call.

## Prerequisites

- A corpus that implements `CorpusProtocol` (for example, `DirectoryCorpus` or `AttuneHelpCorpus`)
- Optional: a custom `RetrieverProtocol` implementation, a `QueryExpander`, or an `LLMReranker`

## Steps

1. **Instantiate `RagPipeline`.**
   Pass your corpus and any optional components to the constructor:

   ```python
   from attune_rag import RagPipeline, DirectoryCorpus

   corpus = DirectoryCorpus("path/to/docs")
   pipeline = RagPipeline(corpus=corpus)
   ```

   Pass a `retriever`, `expander`, or `reranker` argument if you need non-default retrieval or reranking behavior.

2. **Choose how to run the pipeline.**

   - To retrieve context and build an augmented prompt *without* calling an LLM, use `RagPipeline.run`:

     ```python
     result = pipeline.run(query="How do I configure workflows?", k=3)
     print(result.augmented_prompt)
     ```

   - To retrieve context *and* generate a response from an LLM in one step, use `RagPipeline.run_and_generate`:

     ```python
     answer, result = pipeline.run_and_generate(
         query="How do I configure workflows?",
         provider="openai",
         k=3,
         model="gpt-4o",
         prompt_variant="citation",
     )
     print(answer)
     ```

3. **Inspect the `RagResult`.**
   Both methods return a `RagResult` dataclass. Check the fields that matter for your use case:

   | Field | What it tells you |
   |---|---|
   | `augmented_prompt` | The fully assembled prompt sent to the LLM |
   | `citation` | A `CitationRecord` with source provenance |
   | `confidence` | Retrieval confidence score |
   | `fallback_used` | `True` if no grounding context was found in the corpus |
   | `elapsed_ms` | Total pipeline latency in milliseconds |
   | `context` | Raw retrieved context text |
   | `claim_citations` | Per-claim `ClaimCitation` tuples |
   | `used_native_citations` | `True` if the LLM's native citation format was used |

4. **Handle the fallback case.**
   When `result.fallback_used` is `True`, the pipeline found no grounding context. The prompt instructs the LLM to answer honestly and avoid inventing APIs or commands. Decide in your application whether to surface a warning to the user or suppress the response.

5. **Format citations for display.**
   Use `format_citations_markdown(result.citation)` to render source citations, or `format_claim_citations_markdown(result.claim_citations)` to render per-claim citations.

## Verify success

A successful run produces a `RagResult` where:

- `augmented_prompt` is a non-empty string
- `elapsed_ms` is greater than `0`
- `fallback_used` is `False` (confirming the corpus returned at least one relevant document)
- `citation` contains at least one cited source

## Key files

- `src/attune_rag/pipeline.py` — `RagPipeline` and `RagResult` definitions
- `src/attune_rag/__init__.py` — public exports including `PROMPT_VARIANTS` and citation formatting helpers
