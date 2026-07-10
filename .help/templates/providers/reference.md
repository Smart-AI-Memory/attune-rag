---
type: reference
name: providers-reference
feature: providers
depth: reference
generated_at: 2026-07-10T13:05:07.856468+00:00
source_hash: 756ec8cdf5db7cbd88c6a1a079b164855514d934f8d3cafa3743d4318f0339ac
status: generated
---

# Providers reference

Optional async LLM provider adapters. Use these classes to send prompts to Claude or Gemini and receive text or cited responses. Each provider lazy-imports its SDK, so the core package installs without requiring any provider-specific dependency. Install `attune-rag[claude]` or `attune-rag[gemini]` to enable the corresponding provider.

## Classes

| Class | Description |
|-------|-------------|
| `CitationDocument` | One source document passed to a citations-capable provider. |
| `CitedResponse` | Response from a citations-capable provider. |
| `LLMProvider` | An async LLM provider that consumes a prompt and returns text. |
| `ClaudeProvider` | Thin async wrapper over Anthropic's Messages API. |
| `GeminiProvider` | Thin async wrapper over Google's genai models API. |

### CitationDocument

[dataclass]

| Field | Type | Default |
|-------|------|---------|
| `title` | `str` | — |
| `text` | `str` | — |

### CitedResponse

[dataclass]

| Field | Type | Default |
|-------|------|---------|
| `text` | `str` | — |
| `claim_citations` | `tuple[ClaimCitation, ...]` | — |

### LLMProvider

Protocol defining the async LLM provider interface.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Consume a prompt and return generated text. |
| `generate_with_citations` | `self, documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` | Answer a query against a list of source documents and return a cited response. |

### ClaudeProvider

Thin async wrapper over Anthropic's Messages API. Requires `attune-rag[claude]`.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `self, api_key: str \| None = None, client: AsyncAnthropic \| None = None` | `None` | Initialize the provider with an API key or a pre-built `AsyncAnthropic` client. |
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Consume a prompt and return generated text. |
| `generate_with_citations` | `self, documents: list[CitationDocument], query: str, system: str \| None = None, model: str \| None = None, max_tokens: int = 2048` | `CitedResponse` | Answer a query against a list of source documents and return a cited response. |

### GeminiProvider

Thin async wrapper over Google's genai models API. Requires `attune-rag[gemini]`.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `self, api_key: str \| None = None, client: GenAIClient \| None = None` | `None` | Initialize the provider with an API key or a pre-built `GenAIClient` client. |
| `generate` | `self, prompt: str, model: str \| None = None, max_tokens: int = 2048, cached_prefix: str \| None = None` | `str` | Consume a prompt and return generated text. |

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `list_available` | — | `list[str]` | Return the names of providers whose SDKs are importable. |
| `get_provider` | `name: str, **kwargs: Any` | `LLMProvider` | Return an instance of the named provider. |
| `create_message` | `client: Any, kwargs: dict[str, Any]` | `Any` | Dispatch a Messages API call, routing fable models to the beta namespace. |

### Raises

| Function | Raises | Message |
|----------|--------|---------|
| `get_provider` | `ValueError` | `'Unknown provider {...}. Known providers: {...}.'` |
| `create_message` | `ModelRefusalError` | `'model {...} refused the request (the entire server-side fallback chain refused)'` |

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `__all__` | `list` | `{'LLMProvider', 'list_available', 'get_provider'}` |
| `_RETENTION_HINT` | `str` | `"claude-fable-5 requires >=30-day org data retention - check the org's retention configuration before debugging the payload."` |

## Source files

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/openai.py`
- `src/attune_rag/providers/gemini.py`

## Tags

`providers`, `llm`, `claude`, `openai`, `gemini`
