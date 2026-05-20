---
type: reference
name: providers-reference
feature: providers
depth: reference
generated_at: 2026-05-20T03:27:22.832799+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Providers reference

Async LLM provider adapters built around the `LLMProvider` protocol. Each adapter lazy-imports its SDK so the core package installs without optional dependencies. `ClaudeProvider` requires the `attune-rag[claude]` extra; `GeminiProvider` requires `attune-rag[gemini]`.

## Classes

| Class | Description |
|-------|-------------|
| `CitationDocument` | One source document passed to a citations-capable provider. |
| `CitedResponse` | Response from a citations-capable provider. |
| `LLMProvider` | An async LLM provider that consumes a prompt and returns text. |
| `ClaudeProvider` | Thin async wrapper over Anthropic's Messages API. |
| `GeminiProvider` | Thin async wrapper over Google's genai models API. |

### `CitationDocument`

`[dataclass]`

| Field | Type | Default |
|-------|------|---------|
| `title` | `str` | — |
| `text` | `str` | — |

### `CitedResponse`

`[dataclass]`

| Field | Type | Default |
|-------|------|---------|
| `text` | `str` | — |
| `claim_citations` | `tuple[ClaimCitation, ...]` | — |

### `LLMProvider`

Protocol defining the interface for all async LLM providers.

| Method | Parameters | Returns |
|--------|------------|---------|
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` |
| `generate_with_citations` | `self, documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` |

### `ClaudeProvider`

Thin async wrapper over Anthropic's Messages API. Requires `attune-rag[claude]`.

| Method | Parameters | Returns |
|--------|------------|---------|
| `__init__` | `self, api_key: str \| None = None, client: AsyncAnthropic \| None = None` | `None` |
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` |
| `generate_with_citations` | `self, documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` |

### `GeminiProvider`

Thin async wrapper over Google's genai models API. Requires `attune-rag[gemini]`.

| Method | Parameters | Returns |
|--------|------------|---------|
| `__init__` | `self, api_key: str \| None = None, client: GenAIClient \| None = None` | `None` |
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` |

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `list_available` | — | `list[str]` | Return the names of providers whose SDKs are importable. |
| `get_provider` | `name: str, **kwargs` | `LLMProvider` | Return an instance of the named provider. |

### Raises

| Function | Raises | Message |
|----------|--------|---------|
| `get_provider` | `ValueError` | `'Unknown provider {...}. Known providers: {...}.'` |

## Tags

`providers`, `llm`, `claude`, `gemini`
