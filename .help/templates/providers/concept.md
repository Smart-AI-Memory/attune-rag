---
type: concept
name: providers-concept
feature: providers
depth: concept
generated_at: 2026-05-20T03:27:22.821975+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Providers

## How providers work

Providers are optional, installable adapters that connect attune-rag to an external LLM API through a common async interface — the `LLMProvider` protocol — so the rest of the codebase never depends directly on a specific SDK.

Each concrete provider is a thin async wrapper that translates `LLMProvider` method calls into the corresponding API requests:

- **`ClaudeProvider`** wraps Anthropic's Messages API. Install it with the `attune-rag[claude]` extra.
- **`GeminiProvider`** wraps Google's genai models API. Install it with the `attune-rag[gemini]` extra.

Because each provider's SDK is imported lazily, installing attune-rag without any extras succeeds cleanly. You can call `list_available()` at runtime to see which providers have importable SDKs in the current environment, and `get_provider(name, **kwargs)` to retrieve a ready-to-use instance by name.

### The `LLMProvider` protocol

`LLMProvider` declares two async methods that concrete providers implement:

- **`generate(prompt, model, max_tokens, cached_prefix)`** — Sends a plain prompt and returns the model's response as a string. `cached_prefix` lets you mark a portion of the prompt for prompt-caching where the API supports it.
- **`generate_with_citations(documents, query, system, model, max_tokens)`** — Sends a list of source documents alongside a query and returns a `CitedResponse` that pairs the answer text with structured claim-level citations.

### Citation data structures

Two dataclasses support the citations workflow:

- **`CitationDocument`** — A single source document with a `title` (str) and `text` (str) field, passed as a list to `generate_with_citations`.
- **`CitedResponse`** — The structured result returned by `generate_with_citations`, containing the response `text` and a tuple of `ClaimCitation` objects that map individual claims back to their source documents.

## Integration points

Other parts of the codebase interact with providers through these interfaces:

| Interface | Purpose |
|-----------|---------|
| `LLMProvider` | Protocol that all providers implement; the only type the rest of the codebase needs to reference. |
| `CitationDocument` | Input type for citation-aware generation; construct one per source document you want the model to reason over. |
| `CitedResponse` | Output type carrying both the generated text and structured claim citations. |
| `ClaudeProvider` | Concrete implementation for Anthropic; requires `attune-rag[claude]`. |
| `GeminiProvider` | Concrete implementation for Google; requires `attune-rag[gemini]`. |
