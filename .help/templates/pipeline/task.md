---
type: task
feature: pipeline
depth: task
generated_at: 2026-04-23T03:32:53.260013+00:00
source_hash: 65f24abb9bb5f4301d29cbd0c7d716a93bfe027e33389ceb15135635b6d7a679
status: generated
---

# Work with pipeline

Use RagPipeline when you need to combine document retrieval, prompt augmentation, and LLM generation into a single workflow with built-in citation tracking.

## Prerequisites

- Access to a corpus (CorpusProtocol implementation)
- A retriever (RetrieverProtocol implementation)
- An LLM provider for text generation (optional)

## Set up the pipeline

1. **Initialize RagPipeline with your components:**
   ```python
   from attune_rag import RagPipeline, DirectoryCorpus, KeywordRetriever

   corpus = DirectoryCorpus("./docs")
   retriever = KeywordRetriever()
   pipeline = RagPipeline(corpus=corpus, retriever=retriever)
   ```

2. **Run retrieval and prompt assembly:**
   ```python
   result = pipeline.run(
       query="How do I configure authentication?",
       k=3,  # number of documents to retrieve
       prompt_variant="citation"  # or other available variants
   )
   ```

3. **Access the results:**
   ```python
   print(f"Augmented prompt: {result.augmented_prompt}")
   print(f"Citation: {result.citation}")
   print(f"Confidence: {result.confidence}")
   print(f"Fallback used: {result.fallback_used}")
   ```

## Generate responses with an LLM

1. **Run the full pipeline with text generation:**
   ```python
   response, rag_result = pipeline.run_and_generate(
       query="How do I configure authentication?",
       provider="openai",  # or your LLM provider
       model="gpt-4",
       max_tokens=2048
   )
   ```

2. **Handle fallback scenarios:**
   ```python
   if rag_result.fallback_used:
       print("No relevant context found - LLM answered from general knowledge")
   else:
       print(f"Answer grounded in {len(rag_result.context)} characters of context")
   ```

## Extend pipeline behavior

1. **Create a custom pipeline subclass:**
   ```python
   class CustomRagPipeline(RagPipeline):
       def run(self, query: str, k: int = 3, prompt_variant: str = 'citation') -> RagResult:
           # Add custom preprocessing
           processed_query = self._preprocess_query(query)
           return super().run(processed_query, k, prompt_variant)
   ```

2. **Add custom prompt variants:**
   ```python
   # Check available variants
   from attune_rag import PROMPT_VARIANTS
   print(PROMPT_VARIANTS.keys())
   ```

## Verify pipeline setup

Run a test query and confirm you receive a RagResult with:
- Non-empty `augmented_prompt`
- Valid `citation` with source information
- Confidence score between 0 and 1
- Reasonable `elapsed_ms` timing

## Key files

- `src/attune_rag/pipeline.py` — Core RagPipeline and RagResult classes
- `src/attune_rag/__init__.py` — Public API exports
