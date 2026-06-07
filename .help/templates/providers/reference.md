---
type: reference
name: providers-reference
feature: providers
depth: reference
generated_at: 2026-06-07T07:13:23.414944+00:00
source_hash: ab8cfd02877bb1491251eca997f80585ed29819de7e9c31ef4d86c7835dc2891
status: generated
---

# Providers reference

Use the `providers` package to send prompts to LLM backends and retrieve cited responses. `LLMProvider` defines the async protocol; `ClaudeProvider` and `GeminiProvider` are concrete implementations that lazy-import their SDKs, so the core package installs without requiring any provider SDK.

## Module functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `list_available` | — | `list[str]` | Returns the names of providers whose SDKs are importable. |
| `get_provider` | `name: str, **kwargs: Any` | `LLMProvider` | Returns an instance of the named provider. |

### Raises

| Function | Exception | Message |
|----------|-----------|---------|
| `get_provider` | `ValueError` | `'Unknown provider {...}. Known providers: {...}.'` |

## Classes

| Class | Description |
|-------|-------------|
| `CitationDocument` | One source document passed to a citations-capable provider. |
| `CitedResponse` | Response from a citations-capable provider. |
| `LLMProvider` | An async LLM provider that consumes a prompt and returns text. |
| `ClaudeProvider` | Thin async wrapper over Anthropic's Messages API. Requires the `attune-rag[claude]` extra. |
| `GeminiProvider` | Thin async wrapper over Google's genai models API. Requires the `attune-rag[gemini]` extra. |

### `CitationDocument`

`[dataclass]` — One source document passed to a citations-capable provider.

| Field | Type | Default |
|-------|------|---------|
| `title` | `str` | — |
| `text` | `str` | — |

### `CitedResponse`

`[dataclass]` — Response from a citations-capable provider.

| Field | Type | Default |
|-------|------|---------|
| `text` | `str` | — |
| `claim_citations` | `tuple[ClaimCitation, ...]` | — |

### `LLMProvider`

Async protocol that both concrete providers implement.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Sends a prompt and returns the model's text response. |
| `generate_with_citations` | `documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` | Queries the model against a document list and returns a response with claim-level citations. |

### `ClaudeProvider`

Thin async wrapper over Anthropic's Messages API. Requires the `attune-rag[claude]` extra.

**Constructor**

| Parameter | Type | Default |
|-----------|------|---------|
| `api_key` | `str \| None` | `None` |
| `client` | `AsyncAnthropic \| None` | `None` |

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Sends a prompt and returns the model's text response. |
| `generate_with_citations` | `documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` | Queries the model against a document list and returns a response with claim-level citations. |

### `GeminiProvider`

Thin async wrapper over Google's genai models API. Requires the `attune-rag[gemini]` extra.

**Constructor**

| Parameter | Type | Default |
|-----------|------|---------|
| `api_key` | `str \| None` | `None` |
| `client` | `GenAIClient \| None` | `None` |

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Sends a prompt and returns the model's text response. |

## Source files

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/gemini.py`

## Tags

`providers`, `llm`, `claude`, `gemini`
