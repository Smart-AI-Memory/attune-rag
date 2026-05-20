---
type: quickstart
name: providers-quickstart
feature: providers
depth: quickstart
generated_at: 2026-05-20T03:27:22.849567+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Quickstart: providers

`attune-rag` ships optional LLM provider adapters for Claude and Gemini. Each adapter lazy-imports its SDK, so the core package installs without requiring either. Use `list_available()` to confirm which providers are ready on your system, then call `get_provider()` to get a running instance.

```python
from attune_rag.providers import list_available
print(list_available())
# Example output: ['claude', 'gemini']
```

## Prerequisites

- `attune-rag` installed in your environment
- At least one provider SDK installed:
  - Claude: `pip install attune-rag[claude]`
  - Gemini: `pip install attune-rag[gemini]`
- A valid API key for the provider you want to use

## Steps

1. **Check which providers are available.**

   ```python
   from attune_rag.providers import list_available

   print(list_available())
   # ['claude']  ← only providers whose SDKs are importable appear here
   ```

2. **Instantiate a provider.** Pass the provider name to `get_provider()`. Supply your API key as a keyword argument; if you omit it, the provider reads from its default environment variable.

   ```python
   from attune_rag.providers import get_provider

   provider = get_provider("claude", api_key="sk-ant-...")
   ```

   If you pass an unrecognised name, `get_provider()` raises `ValueError: Unknown provider {...}. Known providers: {...}.`

3. **Generate a response.**

   ```python
   import asyncio

   response = asyncio.run(
       provider.generate("Explain retrieval-augmented generation in one sentence.")
   )
   print(response)
   # Retrieval-augmented generation combines a retrieval step with a language
   # model so answers are grounded in specific documents rather than only
   # parametric knowledge.
   ```

4. **Generate a response with citations** *(Claude only).* Pass a list of `CitationDocument` objects and a query to `generate_with_citations()`. The returned `CitedResponse` contains the answer text and a tuple of claim-level citations.

   ```python
   from attune_rag.providers import get_provider
   from attune_rag.providers.base import CitationDocument

   provider = get_provider("claude", api_key="sk-ant-...")
   docs = [
       CitationDocument(title="RAG overview", text="RAG grounds LLM outputs in retrieved documents."),
   ]
   result = asyncio.run(
       provider.generate_with_citations(docs, query="What does RAG do?")
   )
   print(result.text)
   print(result.claim_citations)
   ```

## Next

Read the `LLMProvider` protocol reference to learn how to implement your own adapter that works anywhere `attune-rag` accepts a provider.
