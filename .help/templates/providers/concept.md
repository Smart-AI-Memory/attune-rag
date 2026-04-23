---
type: concept
feature: providers
depth: concept
generated_at: 2026-04-23T03:35:30.408351+00:00
source_hash: e5294788f63c581d926a7a1cd4c23892c9070bf3d23f17e689e2a35655d8b56e
status: generated
---

# Providers

Providers are LLM adapters that expose a common async interface for generating text from different AI services.

## Core abstraction

All providers implement the `LLMProvider` protocol, which defines a single method:

```python
async def generate(prompt: str, model: str | None = None, max_tokens: int = 2048) -> str
```

This uniform interface lets you swap between Claude, OpenAI, and Gemini without changing your application code. Each provider handles the specifics of its underlying SDK while presenting the same async text generation capability.

## Available implementations

The module includes three concrete providers:

- **ClaudeProvider** — wraps Anthropic's Messages API
- **OpenAIProvider** — wraps OpenAI's chat completions API
- **GeminiProvider** — wraps Google's generative AI models

Each provider lazy-imports its SDK, so the core `attune-rag` package installs cleanly without pulling in dependencies you don't need. Install the extras for providers you want:

- `attune-rag[claude]` for ClaudeProvider
- `attune-rag[openai]` for OpenAIProvider
- `attune-rag[gemini]` for GeminiProvider

## Dynamic provider discovery

The module provides runtime discovery through two functions:

- `list_available()` returns names of providers whose SDKs are importable
- `get_provider(name, **kwargs)` instantiates the named provider with your configuration

This pattern lets you configure providers dynamically based on what's actually installed, rather than hardcoding imports that might fail.

## Initialization patterns

Each provider accepts either an API key string or a pre-configured client instance. This flexibility supports both simple setups (just pass your key) and advanced scenarios where you need custom client configuration:

```python
# Simple: provider handles client creation
claude = ClaudeProvider(api_key="your-key")

# Advanced: you manage the client
client = AsyncAnthropic(api_key="your-key", timeout=30)
claude = ClaudeProvider(client=client)
```
