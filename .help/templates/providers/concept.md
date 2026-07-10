---
type: concept
name: providers-concept
feature: providers
depth: concept
generated_at: 2026-07-10T13:05:07.848150+00:00
source_hash: 756ec8cdf5db7cbd88c6a1a079b164855514d934f8d3cafa3743d4318f0339ac
status: generated
---

# Providers

A provider is an async adapter that wraps a third-party LLM SDK — Claude or Gemini — behind a single, uniform interface so the rest of attune-rag can generate text and citations without knowing which model is in use.

## How providers fit together

Every provider implements the `LLMProvider` protocol, which defines two async methods:

- `generate(prompt, model, max_tokens, cached_prefix)` — sends a prompt and returns a plain text string.
- `generate_with_citations(documents, query, system, model, max_tokens)` — sends a list of source documents alongside a query and returns a `CitedResponse` that pairs the answer text with structured `ClaimCitation` references.

The two concrete implementations sit behind optional install extras so the core package stays lightweight:

| Provider | Install extra | Wraps |
|---|---|---|
| `ClaudeProvider` | `attune-rag[claude]` | Anthropic's Messages API (`AsyncAnthropic`) |
| `GeminiProvider` | `attune-rag[gemini]` | Google's genai models API (`GenAIClient`) |

Each provider lazy-imports its SDK, which means importing attune-rag itself never fails just because the Anthropic or Google library is absent.

## Citation data model

When you call `generate_with_citations`, the provider expects input as a list of `CitationDocument` objects — each holding a `title` and `text` — and returns a `CitedResponse` with two fields:

- `text` — the generated answer.
- `claim_citations` — a tuple of `ClaimCitation` values that map specific claims in the answer back to the source documents.

This structure lets callers display grounded, traceable responses without parsing raw model output.

## Provider discovery and instantiation

Two module-level functions handle provider selection at runtime:

- `list_available() -> list[str]` — returns the names of providers whose SDKs are currently importable, so you can check what is usable before committing to one.
- `get_provider(name, **kwargs) -> LLMProvider` — returns a live provider instance by name; raises `ValueError` if the name is unknown.

Both `ClaudeProvider` and `GeminiProvider` accept either an API key string or a pre-constructed async client, which makes them straightforward to inject in tests.

## When this matters

You interact with providers any time attune-rag needs to call an LLM. If you are adding a new model backend, you implement `LLMProvider` and register it so `get_provider` can return it. If you are debugging a generation failure with `claude-fable-5`, note that the model requires at least 30-day organisational data retention — check the org's retention configuration before inspecting the request payload.
