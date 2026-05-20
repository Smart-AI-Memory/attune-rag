---
type: faq
name: providers-faq
feature: providers
depth: faq
generated_at: 2026-05-20T03:27:22.847297+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Providers FAQ

## What is the providers module?

The providers module contains optional async adapters that connect your code to LLM backends. It defines the `LLMProvider` protocol and ships two implementations: `ClaudeProvider` (Anthropic) and `GeminiProvider` (Google). Each adapter lazy-imports its SDK, so the core `attune-rag` package installs without requiring any LLM SDK.

## Which providers are available?

`ClaudeProvider` and `GeminiProvider`. To check which ones are importable in your current environment, call `list_available()` — it returns the names of providers whose SDKs are already installed.

## How do I install a specific provider?

Install the matching extra:

- Claude: `pip install attune-rag[claude]`
- Gemini: `pip install attune-rag[gemini]`

## How do I get a provider instance?

Call `get_provider(name, **kwargs)`, where `name` is `"claude"` or `"gemini"`. The function returns a ready-to-use `LLMProvider` instance. It raises `ValueError` if you pass an unrecognised name, and the error message lists the known providers.

## What can I do with a provider?

Every `LLMProvider` exposes two async methods:

- `generate(prompt, model, max_tokens, cached_prefix)` — sends a prompt and returns a plain text response.
- `generate_with_citations(documents, query, system, model, max_tokens)` — sends a list of `CitationDocument` objects and returns a `CitedResponse` containing the answer text and a tuple of `ClaimCitation` objects.

Note: `GeminiProvider` implements `generate` but not `generate_with_citations`.

## What are `CitationDocument` and `CitedResponse`?

`CitationDocument` is a dataclass with `title: str` and `text: str` fields that you pass to `generate_with_citations` as the source documents. `CitedResponse` is what you get back: a `text: str` answer and `claim_citations: tuple[ClaimCitation, ...]` mapping claims in the answer to their sources.

## Can I pass my own SDK client instead of an API key?

Yes. Both `ClaudeProvider` and `GeminiProvider` accept either an `api_key` string or a pre-configured async client object (`AsyncAnthropic` or `GenAIClient`) in their constructors. If you pass neither, the SDK's default credential resolution applies.

## How do I debug a provider that isn't working?

1. Call `list_available()` to confirm the provider's SDK is importable.
2. Run the related tests: `pytest -k "providers" -v`.
3. If the tests pass but your code still fails, add a `logger.debug` statement just before your `generate` or `generate_with_citations` call and re-run with logging enabled to inspect the inputs.

## Where are the source files?

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/gemini.py`

**Tags:** `providers`, `llm`, `claude`, `gemini`
