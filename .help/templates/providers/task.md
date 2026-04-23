---
type: task
feature: providers
depth: task
generated_at: 2026-04-23T03:35:41.924612+00:00
source_hash: e5294788f63c581d926a7a1cd4c23892c9070bf3d23f17e689e2a35655d8b56e
status: generated
---

# Work with providers

Use providers when you need to integrate LLM services like Claude, OpenAI, or Gemini into your application through a consistent async interface.

## Prerequisites

- Access to the project source code
- API key for your chosen provider (Claude, OpenAI, or Gemini)
- Understanding of async Python programming

## Configure a provider

1. **Check available providers.**
   Run `list_available()` to see which provider SDKs are installed:
   ```python
   from attune_rag.providers import list_available
   available = list_available()
   print(available)  # ['claude', 'openai', 'gemini']
   ```

2. **Install the provider SDK.**
   Install the extra dependency for your chosen provider:
   - For Claude: `pip install attune-rag[claude]`
   - For OpenAI: `pip install attune-rag[openai]`
   - For Gemini: `pip install attune-rag[gemini]`

3. **Create a provider instance.**
   Use `get_provider()` with your API key:
   ```python
   from attune_rag.providers import get_provider
   provider = get_provider('claude', api_key='your-api-key')
   ```

4. **Generate text.**
   Call the `generate()` method with your prompt:
   ```python
   response = await provider.generate(
       prompt="Explain quantum computing",
       model="claude-3-sonnet-20240229",  # optional
       max_tokens=1000  # optional, defaults to 2048
   )
   print(response)
   ```

## Verify the setup

You successfully configured a provider when:
- `list_available()` returns your provider name
- `get_provider()` creates an instance without errors
- `generate()` returns text responses to your prompts

## Key files

- `src/attune_rag/providers/__init__.py` — Main entry points
- `src/attune_rag/providers/base.py` — LLMProvider protocol
- `src/attune_rag/providers/claude.py` — Claude implementation
- `src/attune_rag/providers/openai.py` — OpenAI implementation
- `src/attune_rag/providers/gemini.py` — Gemini implementation
