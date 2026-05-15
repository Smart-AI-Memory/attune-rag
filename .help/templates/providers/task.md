---
type: task
name: providers-task
feature: providers
depth: task
generated_at: 2026-05-15T20:02:55.654090+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Work with providers

Use providers when you need to connect attune-rag to an LLM backend — selecting, instantiating, and calling Claude or Gemini through a shared `LLMProvider` interface so the rest of your application stays provider-agnostic.

## Prerequisites

- An API key for the provider you want to use (Claude or Gemini)
- The matching optional SDK installed:
  - Claude: `pip install attune-rag[claude]`
  - Gemini: `pip install attune-rag[gemini]`
- Read access to `src/attune_rag/providers/`

## Steps

1. **Check which providers are available in your environment.**
   Call `list_available()` to get the names of providers whose SDKs are currently importable:

   ```python
   from attune_rag.providers import list_available
   print(list_available())
   ```

2. **Retrieve a provider instance.**
   Call `get_provider(name, **kwargs)` with the provider name returned by `list_available()`. Pass your API key as a keyword argument if you are not using an environment variable:

   ```python
   from attune_rag.providers import get_provider
   provider = get_provider("claude", api_key="your-api-key")
   ```

   If you pass an unknown name, `get_provider` raises `ValueError: Unknown provider {…}. Known providers: {…}.`

3. **Generate a response.**
   Call `provider.generate()` with your prompt. Supply `model` and `max_tokens` if you want to override the defaults:

   ```python
   response = await provider.generate(
       prompt="Summarise the following text: ...",
       model="claude-opus-4-5",
       max_tokens=1024,
   )
   print(response)
   ```

4. **Generate a response with citations (Claude only).**
   If you need cited claims, build a list of `CitationDocument` objects and call `generate_with_citations()`:

   ```python
   from attune_rag.providers import CitationDocument
   docs = [
       CitationDocument(title="Source A", text="..."),
       CitationDocument(title="Source B", text="..."),
   ]
   cited = await provider.generate_with_citations(
       documents=docs,
       query="What does Source A say about X?",
   )
   print(cited.text)
   print(cited.claim_citations)
   ```

5. **Run the provider tests to confirm everything works.**
   ```
   pytest -k "providers"
   ```

## Verify success

`list_available()` returns at least one provider name, `get_provider()` returns an object without raising `ValueError`, and `provider.generate()` returns a non-empty string. All `pytest -k "providers"` tests pass.

## Key files

| File | Purpose |
|---|---|
| `src/attune_rag/providers/__init__.py` | Public API: `list_available`, `get_provider` |
| `src/attune_rag/providers/base.py` | `LLMProvider` protocol definition |
| `src/attune_rag/providers/claude.py` | `ClaudeProvider` implementation |
| `src/attune_rag/providers/gemini.py` | `GeminiProvider` implementation |
