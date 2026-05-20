---
type: task
name: providers-task
feature: providers
depth: task
generated_at: 2026-05-20T03:27:22.827948+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Work with providers

Use an LLM provider when you need to send prompts to Claude or Gemini and receive generated text — or cited responses — through a consistent async interface without coupling your code to a specific SDK.

## Prerequisites

- An API key for your chosen provider (Anthropic or Google)
- The matching optional dependency installed:
  - Claude: `pip install attune-rag[claude]`
  - Gemini: `pip install attune-rag[gemini]`

## Steps

1. **Check which providers are available in your environment.**
   Call `list_available()` to see which provider SDKs are currently importable:

   ```python
   from attune_rag.providers import list_available

   print(list_available())  # e.g. ["claude", "gemini"]
   ```

2. **Instantiate a provider.**
   Call `get_provider()` with the provider name and your API key. If the name is not recognised, it raises a `ValueError` listing the known providers.

   ```python
   from attune_rag.providers import get_provider

   provider = get_provider("claude", api_key="YOUR_API_KEY")
   # or
   provider = get_provider("gemini", api_key="YOUR_API_KEY")
   ```

   Alternatively, instantiate a provider class directly if you want to supply a pre-configured SDK client:

   ```python
   from attune_rag.providers.claude import ClaudeProvider
   from anthropic import AsyncAnthropic

   provider = ClaudeProvider(client=AsyncAnthropic(api_key="YOUR_API_KEY"))
   ```

3. **Generate a response from a prompt.**
   Call `generate()` with a prompt string. Optionally specify `model`, `max_tokens` (default `2048`), or a `cached_prefix` for prompt caching:

   ```python
   text = await provider.generate(
       prompt="Summarise the history of Rome.",
       model="claude-opus-4-5",
       max_tokens=512,
   )
   print(text)
   ```

4. **Generate a cited response from source documents** *(Claude only).*
   Build a list of `CitationDocument` objects and call `generate_with_citations()`. The returned `CitedResponse` contains the generated text and a tuple of claim-level citations.

   ```python
   from attune_rag.providers import CitationDocument

   documents = [
       CitationDocument(title="Rome Wikipedia", text="Rome was founded in 753 BC..."),
       CitationDocument(title="Ancient History Encyclopedia", text="The Roman Republic began..."),
   ]

   response = await provider.generate_with_citations(
       documents=documents,
       query="When was Rome founded?",
       system="Answer only from the provided documents.",
   )

   print(response.text)
   print(response.claim_citations)
   ```

   > **Note:** `GeminiProvider` does not implement `generate_with_citations`.

## Verify success

- `list_available()` returns a non-empty list that includes your target provider.
- `generate()` returns a non-empty string without raising an exception.
- `generate_with_citations()` returns a `CitedResponse` whose `claim_citations` tuple contains at least one entry referencing one of your input documents.
