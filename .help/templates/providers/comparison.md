---
type: comparison
name: providers-comparison
feature: providers
depth: comparison
generated_at: 2026-05-20T03:27:22.857254+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Comparison: ClaudeProvider vs GeminiProvider

## Context

`attune-rag` ships two concrete LLM provider adapters — `ClaudeProvider` and `GeminiProvider` — both implementing the `LLMProvider` protocol. Neither SDK is imported at install time; each adapter lazy-imports its dependency so the core package stays lightweight. You install only the adapter you need via an optional extra.

## Feature comparison

| Capability | `ClaudeProvider` | `GeminiProvider` |
|---|---|---|
| **Install extra** | `attune-rag[claude]` | `attune-rag[gemini]` |
| **Underlying SDK** | `anthropic` (`AsyncAnthropic`) | `google-genai` (`GenAIClient`) |
| **`generate()`** | ✅ | ✅ |
| **`generate_with_citations()`** | ✅ Returns `CitedResponse` with `claim_citations` | ❌ Not implemented |
| **`cached_prefix` support in `generate()`** | ✅ | ✅ |
| **Custom client injection** | ✅ Pass an `AsyncAnthropic` instance | ✅ Pass a `GenAIClient` instance |
| **API key override** | ✅ `api_key` constructor argument | ✅ `api_key` constructor argument |
| **Default `max_tokens`** | 2048 | 2048 |

### `generate_with_citations()` in detail

`ClaudeProvider` is the only adapter that implements `generate_with_citations()`. It accepts a list of `CitationDocument` objects (each with a `title` and `text`), a `query` string, and an optional `system` prompt, and returns a `CitedResponse` containing:

- `text` — the generated answer
- `claim_citations` — a tuple of `ClaimCitation` objects linking specific claims back to source documents

`GeminiProvider` does not implement `generate_with_citations()`. Calling it through the `LLMProvider` protocol will raise `NotImplementedError`.

## Discovering and instantiating providers at runtime

Use the two module-level helpers when you want provider selection to be dynamic:

```python
from attune_rag.providers import list_available, get_provider

# Returns only providers whose SDKs are currently importable, e.g. ['claude', 'gemini']
available = list_available()

# Instantiate by name; raises ValueError for unknown names
provider = get_provider("claude", api_key="sk-...")
```

`get_provider()` raises `ValueError` with the message `'Unknown provider {name}. Known providers: {known}.'` if you pass a name that is not registered.

## When NOT to use these adapters directly

- **You need a provider not listed here.** The source includes a `openai.py` file but no `OpenAIProvider` is exported in `__all__` or documented in the public API. Do not rely on it; check `list_available()` at runtime to confirm what is actually importable in your environment.
- **You need synchronous calls.** Both adapters are async-only. If your codebase is synchronous, wrap calls with `asyncio.run()` or use an orchestration layer above the providers module.
- **You need citation support with Gemini.** `GeminiProvider` does not implement `generate_with_citations()`. Switch to `ClaudeProvider` or implement a custom `LLMProvider` subclass.
- **You are doing one-off exploration.** Instantiating a provider requires a valid API key and network access. A direct SDK call is simpler for throwaway scripts.

## Use X when...

**Use `ClaudeProvider` when** you need citation tracking — it is the only adapter that returns `CitedResponse` with `claim_citations`, making it the right choice for RAG pipelines where you need to attribute generated claims back to source documents.

**Use `GeminiProvider` when** you only need `generate()` and your organisation already uses Google's genai platform, or when you want to avoid the Anthropic SDK dependency.

**Use `get_provider()` with `list_available()` when** you are building a multi-provider application and want to select the adapter at runtime based on which SDK credentials are present in the environment.

**Use a custom `LLMProvider` implementation when** neither adapter fits — the protocol requires only `generate()` and `generate_with_citations()`, so a minimal implementation is straightforward.

## Source files

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/gemini.py`

**Tags:** `providers`, `llm`, `claude`, `gemini`
