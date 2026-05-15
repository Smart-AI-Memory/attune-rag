---
type: concept
name: providers-concept
feature: providers
depth: concept
generated_at: 2026-05-15T20:02:55.650107+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Providers

The `providers` module is a set of optional, swappable adapters that let attune-rag send prompts to an LLM and receive structured responses, without requiring every LLM's SDK to be installed.

## The provider model

At the center is `LLMProvider`, an async protocol that every concrete adapter implements. It defines two methods:

- `generate(prompt, model, max_tokens, cached_prefix)` — sends a prompt and returns a plain text response.
- `generate_with_citations(documents, query, system, model, max_tokens)` — sends a list of source documents alongside a query and returns a `CitedResponse` that pairs the answer text with per-claim citations.

Any code in attune-rag that needs an LLM depends only on `LLMProvider`, not on a specific SDK. You can swap `ClaudeProvider` for `GeminiProvider` without changing the calling code.

## Concrete adapters

Two adapters ship with attune-rag:

| Adapter | Wraps | Install extra |
|---|---|---|
| `ClaudeProvider` | Anthropic's Messages API (`AsyncAnthropic`) | `attune-rag[claude]` |
| `GeminiProvider` | Google's genai models API (`GenAIClient`) | `attune-rag[gemini]` |

Each adapter lazy-imports its SDK, so the core package installs cleanly even when neither extra is present. If you try to use a provider whose SDK is not installed, `list_available()` will simply not include it.

## Selecting a provider at runtime

Two module-level functions handle discovery and instantiation:

- `list_available() -> list[str]` — returns the names of providers whose SDKs are currently importable. Use this to check what is available before committing to a provider.
- `get_provider(name, **kwargs) -> LLMProvider` — returns a ready-to-use instance of the named provider. Raises `ValueError` if the name is unknown, with a message that lists known providers.

A typical usage pattern looks like this: call `list_available()` to confirm the desired provider is present, then pass the name to `get_provider()` and use the returned object anywhere `LLMProvider` is expected.

## Citation data flow

When a provider supports citations, you wrap each source document in a `CitationDocument` (a dataclass with a `title` and `text` field) and pass the list to `generate_with_citations`. The method returns a `CitedResponse` containing the answer `text` and a tuple of `ClaimCitation` objects that map specific claims in the response back to the source documents.

`GeminiProvider` implements only `generate`; citation support is specific to `ClaudeProvider`.

## When this matters

You interact with the providers layer whenever you need to:

- Add a new LLM backend without modifying core attune-rag logic.
- Check at startup which LLMs are available in the current environment.
- Return responses that include traceable citations to source documents.
