---
type: reference
feature: providers
depth: reference
generated_at: 2026-04-23T03:35:51.402594+00:00
source_hash: e5294788f63c581d926a7a1cd4c23892c9070bf3d23f17e689e2a35655d8b56e
status: generated
---

# Providers reference

Optional LLM provider adapters for Anthropic, OpenAI, and Google models.

## Classes

| Class | Description |
|-------|-------------|
| `LLMProvider` | An async LLM provider that consumes a prompt and returns text |
| `ClaudeProvider` | Thin async wrapper over Anthropic's Messages API |
| `OpenAIProvider` | Thin async wrapper over OpenAI's chat completions API |
| `GeminiProvider` | Thin async wrapper over Google's genai models API |

## LLMProvider

Protocol defining the interface for LLM providers.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate` | `prompt: str, model: str | None = None, max_tokens: int = 2048` | `str` | Generate text from prompt using specified model |

## ClaudeProvider

Requires `attune-rag[claude]` extra.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `api_key: str | None = None, client: AsyncAnthropic | None = None` | `None` | Initialize provider with API key or client |
| `generate` | `prompt: str, model: str | None = None, max_tokens: int = 2048` | `str` | Generate text using Claude models |

## OpenAIProvider

Requires `attune-rag[openai]` extra.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `api_key: str | None = None, client: AsyncOpenAI | None = None` | `None` | Initialize provider with API key or client |
| `generate` | `prompt: str, model: str | None = None, max_tokens: int = 2048` | `str` | Generate text using OpenAI models |

## GeminiProvider

Requires `attune-rag[gemini]` extra.

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `api_key: str | None = None, client: GenAIClient | None = None` | `None` | Initialize provider with API key or client |
| `generate` | `prompt: str, model: str | None = None, max_tokens: int = 2048` | `str` | Generate text using Google Gemini models |

## Functions

| Function | Parameters | Returns | Description | Raises |
|----------|------------|---------|-------------|---------|
| `list_available` | | `list[str]` | Return the names of providers whose SDKs are importable | |
| `get_provider` | `name: str, **kwargs` | `LLMProvider` | Return an instance of the named provider | `ValueError` — 'Unknown provider {...}. Known providers: {...}.' |
