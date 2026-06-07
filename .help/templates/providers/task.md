---
type: task
name: providers-task
feature: providers
depth: task
generated_at: 2026-06-07T07:13:23.409771+00:00
source_hash: ab8cfd02877bb1491251eca997f80585ed29819de7e9c31ef4d86c7835dc2891
status: generated
---

# Work with providers

Use providers when you need to generate text or citation-backed responses from an LLM, and you want to swap between Claude and Gemini without changing your application code.

## Prerequisites

- An API key for the provider you want to use (Anthropic for Claude, Google for Gemini)
- The matching optional SDK installed:
  - Claude: `pip install attune-rag[claude]`
  - Gemini: `pip install attune-rag[gemini]`

## Steps

### Check which providers are available

1. Import `list_available` from `providers`:

   ```python
   from providers import list_available

   print(list_available())
   ```

   `list_available()` returns the names of providers whose SDKs are currently importable. If a provider's SDK is not installed, it does not appear in the list.

### Get a provider instance

2. Call `get_provider(name, **kwargs)` with the provider name and any constructor arguments:

   ```python
   from providers import get_provider

   provider = get_provider("claude", api_key="sk-...")
   ```

   Pass `api_key` to authenticate. If you omit it, the provider reads the key from its default environment variable. `get_provider` raises `ValueError` if the name is not recognized, with a message listing the known providers.

### Generate text

3. Call `generate` on the provider instance:

   ```python
   response = await provider.generate(
       prompt="Summarize the history of functional programming.",
       model=None,        # uses the provider's default model
       max_tokens=2048,
   )
   print(response)
   ```

   Pass `cached_prefix` if you want to reuse a long shared context across multiple calls.

### Generate a response with citations

4. Build a list of `CitationDocument` objects, one per source document:

   ```python
   from providers.base import CitationDocument

   documents = [
       CitationDocument(title="SICP", text="...chapter text..."),
       CitationDocument(title="TAPL", text="...chapter text..."),
   ]
   ```

5. Call `generate_with_citations` on a provider that supports it (for example, `ClaudeProvider`):

   ```python
   result = await provider.generate_with_citations(
       documents=documents,
       query="What is a lambda calculus?",
       system="You are a computer science tutor.",
       max_tokens=2048,
   )
   ```

   The method returns a `CitedResponse` with two fields:
   - `text` — the generated answer
   - `claim_citations` — a tuple of `ClaimCitation` objects linking claims in the answer to source documents

## Verify the task worked

- `list_available()` returns a non-empty list that includes your target provider name.
- `get_provider(name)` returns an object without raising `ValueError`.
- `generate(...)` returns a non-empty string.
- For citation calls, `result.text` is non-empty and `result.claim_citations` contains at least one entry when the answer draws on the supplied documents.
