---
type: task
name: providers-task
feature: providers
depth: task
generated_at: 2026-07-10T13:05:07.853010+00:00
source_hash: 756ec8cdf5db7cbd88c6a1a079b164855514d934f8d3cafa3743d4318f0339ac
status: generated
---

# Work with providers

Use the `providers` module when you need to connect attune-rag to a Claude or Gemini backend, check which provider SDKs are installed, or add support for a new LLM provider.

## Prerequisites

- An API key for the provider you want to use (Claude requires `attune-rag[claude]`; Gemini requires `attune-rag[gemini]`)
- Python `pytest` installed to verify your changes

## Steps

1. **Check which providers are available.**
   Call `list_available()` to see which provider SDKs are importable in the current environment:

   ```python
   from attune_rag.providers import list_available
   print(list_available())  # e.g. ['claude', 'gemini']
   ```

2. **Instantiate the provider you need.**
   Call `get_provider(name, **kwargs)` with the provider name returned by `list_available()`. Pass your API key if you are not relying on an environment variable:

   ```python
   from attune_rag.providers import get_provider
   provider = get_provider("claude", api_key="sk-...")
   ```

   If you pass an unrecognised name, `get_provider` raises `ValueError: 'Unknown provider {…}. Known providers: {…}.'`

3. **Generate a response.**
   Call `provider.generate()` with your prompt. Override `model` or `max_tokens` when the defaults do not suit your use case:

   ```python
   text = await provider.generate(
       prompt="Summarise the following text…",
       model="claude-opus-4-5",
       max_tokens=512,
   )
   ```

4. **Generate a response with citations (Claude only).**
   Build a list of `CitationDocument` objects and call `generate_with_citations()`. The returned `CitedResponse` contains the answer text and a tuple of `ClaimCitation` objects that map each claim back to a source document:

   ```python
   from attune_rag.providers import CitationDocument

   docs = [
       CitationDocument(title="RAG overview", text="…"),
       CitationDocument(title="Provider guide", text="…"),
   ]
   response = await provider.generate_with_citations(
       documents=docs,
       query="What is attune-rag?",
   )
   print(response.text)
   print(response.claim_citations)
   ```

5. **Add or modify a provider (optional).**
   Open the relevant file from the list below and implement the `LLMProvider` protocol — `generate()` is required; `generate_with_citations()` is optional. Mirror the error-handling and logging style used in `claude.py` or `gemini.py`, and register the new name inside `get_provider()` in `__init__.py`.

   > **Note:** If you use a `claude-fable-5` model, the org account must have ≥ 30-day data retention configured. Check this before debugging an unexpected refusal — `create_message()` will raise `ModelRefusalError` if the entire server-side fallback chain refuses the request.

6. **Run the tests.**
   Confirm that nothing regressed:

   ```bash
   pytest -k "providers"
   ```

## Key files

| File | Purpose |
|---|---|
| `src/attune_rag/providers/__init__.py` | Public API: `list_available()`, `get_provider()` |
| `src/attune_rag/providers/base.py` | `LLMProvider` protocol definition |
| `src/attune_rag/providers/claude.py` | `ClaudeProvider` + `create_message()` dispatcher |
| `src/attune_rag/providers/gemini.py` | `GeminiProvider` wrapper |

## Verify success

`pytest -k "providers"` passes with no failures or errors. If you added a new provider, `list_available()` returns its name when the corresponding SDK is installed, and `get_provider("<name>")` returns an instance without raising `ValueError`.
