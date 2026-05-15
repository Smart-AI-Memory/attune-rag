---
type: reference
name: providers-reference
feature: providers
depth: reference
generated_at: 2026-05-15T20:02:55.656803+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Providers reference

Use this module to send prompts to Claude or Gemini and receive text or cited responses. `LLMProvider` defines the async protocol; `ClaudeProvider` and `GeminiProvider` are the concrete implementations. Each provider lazy-imports its SDK, so the core package installs without requiring any provider SDK. `ClaudeProvider` requires the `attune-rag[claude]` extra; `GeminiProvider` requires `attune-rag[gemini]`.

## Classes

| Class | Description |
|-------|-------------|
| `CitationDocument` | One source document passed to a citations-capable provider. |
| `CitedResponse` | Response from a citations-capable provider. |
| `LLMProvider` | An async LLM provider that consumes a prompt and returns text. |
| `ClaudeProvider` | Thin async wrapper over Anthropic's Messages API. |
| `GeminiProvider` | Thin async wrapper over Google's genai models API. |

### CitationDocument

`[dataclass]`

| Field | Type | Default |
|-------|------|---------|
| `title` | `str` | — |
| `text` | `str` | — |

### CitedResponse

`[dataclass]`

| Field | Type | Default |
|-------|------|---------|
| `text` | `str` | — |
| `claim_citations` | `tuple[ClaimCitation, ...]` | — |

### LLMProvider

Protocol defining the async interface all providers implement.

**Methods**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Send a prompt and return the model's text response. |
| `generate_with_citations` | `self, documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` | Answer a query grounded in the supplied documents and return the response with citations. |

### ClaudeProvider

Thin async wrapper over Anthropic's Messages API. Requires the `attune-rag[claude]` extra.

**Constructor**

| Parameters | Returns |
|------------|---------|
| `self, api_key: str \| None = None, client: AsyncAnthropic \| None = None` | `None` |

**Methods**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Send a prompt and return the model's text response. |
| `generate_with_citations` | `self, documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` | Answer a query grounded in the supplied documents and return the response with citations. |

### GeminiProvider

Thin async wrapper over Google's genai models API. Requires the `attune-rag[gemini]` extra.

**Constructor**

| Parameters | Returns |
|------------|---------|
| `self, api_key: str \| None = None, client: GenAIClient \| None = None` | `None` |

**Methods**

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Send a prompt and return the model's text response. |

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `list_available` | — | `list[str]` | Return the names of providers whose SDKs are importable. |
| `get_provider` | `name: str, **kwargs` | `LLMProvider` | Return an instance of the named provider. |

### Raises

| Function | Raises | Message |
|----------|--------|---------|
| `get_provider` | `ValueError` | `'Unknown provider {...}. Known providers: {...}.'` |

## Source files

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/openai.py`
- `src/attune_rag/providers/gemini.py`

## Tags

`providers`, `llm`, `claude`, `openai`, `gemini`
