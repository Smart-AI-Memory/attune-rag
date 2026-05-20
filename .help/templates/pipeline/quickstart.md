---
type: quickstart
name: pipeline-quickstart
feature: pipeline
depth: quickstart
generated_at: 2026-05-20T03:20:40.538198+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Quickstart: pipeline

Run a retrieval-augmented generation pipeline end-to-end with `RagPipeline`.

```python
from attune_rag import RagPipeline, DirectoryCorpus

pipeline = RagPipeline(corpus=DirectoryCorpus("docs/"))
answer, result = pipeline.run_and_generate(
    query="How do I configure a workflow?",
    provider="openai",
)
print(answer)
print(f"Confidence: {result.confidence:.2f}  |  Elapsed: {result.elapsed_ms:.0f} ms")
```

Expected output:

```
To configure a workflow, open the attune dashboard and ...
Confidence: 0.87  |  Elapsed: 312 ms
```

## Prerequisites

- Install the package: `pip install attune-rag`
- Set the environment variable for your LLM provider (for example, `OPENAI_API_KEY`)

## Steps

### 1. Build a corpus

Point `DirectoryCorpus` at a folder of documents. The pipeline retrieves grounding context from these files at query time.

```python
from attune_rag import DirectoryCorpus

corpus = DirectoryCorpus("docs/")
```

### 2. Create the pipeline

Pass the corpus to `RagPipeline`. All other arguments — retriever, expander, reranker — are optional and default to built-in implementations.

```python
from attune_rag import RagPipeline

pipeline = RagPipeline(corpus=corpus)
```

### 3. Run a query

Call `run_and_generate` with your query and LLM provider. It returns the generated answer string and a `RagResult` containing provenance metadata.

```python
answer, result = pipeline.run_and_generate(
    query="How do I configure a workflow?",
    provider="openai",
)

print(answer)
print(result.citation)       # CitationRecord with source provenance
print(result.fallback_used)  # True if no grounding context was found
```

If `result.fallback_used` is `True`, the corpus contained no relevant documents for the query. The pipeline still responds, but without grounding context.

### 4. Format citations

Turn the citation provenance into readable Markdown for display or logging.

```python
from attune_rag import format_citations_markdown

print(format_citations_markdown(result.citation))
```

Expected output:

```
**Sources**
1. docs/workflow-config.md — §3 "Configuring a workflow"
2. docs/reference.md — §7 "Workflow parameters"
```

## Next

Read the `RagPipeline` reference page to learn how to swap in a custom retriever or reranker.

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
