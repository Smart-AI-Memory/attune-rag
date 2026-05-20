---
type: troubleshooting
name: providers-troubleshooting
feature: providers
depth: troubleshooting
generated_at: 2026-05-20T03:27:22.844671+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Troubleshoot providers

## Before you start

The `providers` module ships optional LLM adapters for Claude and Gemini. Each adapter lazy-imports its SDK, so the core package installs without them. You must install the appropriate extra before using a provider:

- `ClaudeProvider` → `pip install attune-rag[claude]`
- `GeminiProvider` → `pip install attune-rag[gemini]`

To check which providers are currently importable in your environment, call `list_available()` before instantiating anything.

> **Note:** The source files include an `openai.py` module, but `OpenAIProvider` is not part of the public API (`__all__`) and is not documented. Do not rely on it.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `ValueError: Unknown provider {...}` from `get_provider()` | Run `list_available()` — the provider's SDK is probably not installed. Install the required extra and verify the name matches a value returned by `list_available()`. |
| `ImportError` or `ModuleNotFoundError` on import | The optional SDK is missing. Run `pip install attune-rag[claude]` or `pip install attune-rag[gemini]` as appropriate. |
| `generate()` or `generate_with_citations()` raises an authentication error | Your API key is missing or invalid. Check that the `api_key` argument is set, or that the relevant environment variable (`ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`) is exported in the current shell. |
| `generate_with_citations()` raises `AttributeError` or `NotImplementedError` | `GeminiProvider` does not implement `generate_with_citations()`. Switch to `ClaudeProvider`, or use `generate()` instead. |
| Provider instantiates but every response is truncated | `max_tokens` defaults to `2048`. Pass a higher value explicitly: `provider.generate(prompt, max_tokens=4096)`. |
| Intermittent failures on repeated calls | Check for API-side rate limiting or network timeouts. These originate outside this library; inspect the exception message from the upstream SDK for status codes. |

## Step-by-step diagnosis

1. **Confirm the provider is available.**
   Before anything else, run:
   ```python
   from attune_rag.providers import list_available
   print(list_available())
   ```
   If the provider you need is absent from the output, install its extra (see [Before you start](#before-you-start)) and re-run.

2. **Reproduce the failure with a minimal call.**
   Strip the call down to required arguments only. For `ClaudeProvider`:
   ```python
   import asyncio
   from attune_rag.providers.claude import ClaudeProvider

   provider = ClaudeProvider(api_key="YOUR_KEY")
   result = asyncio.run(provider.generate("Hello"))
   print(result)
   ```
   Confirm the failure still occurs before adding back any surrounding context.

3. **Enable DEBUG logging.**
   The underlying Anthropic and Google SDKs respect standard Python logging. Enable `DEBUG` to surface raw request/response details:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
   Re-run your minimal reproduction and look for authentication errors, malformed payloads, or unexpected model names in the output.

4. **Check the model name and parameters.**
   Both `ClaudeProvider.generate()` and `GeminiProvider.generate()` accept an optional `model` argument. If `model=None`, the provider uses its default. An unrecognized model name typically raises an error from the upstream SDK, not from `attune-rag` itself. Pass the model string explicitly to rule this out.

5. **Run the related tests.**
   ```bash
   pytest -k "providers" -v
   ```
   If a test covers the failing path, check whether it passes. A passing test with a failing integration usually points to an environment difference (API key, installed SDK version, network access).

## Common fixes

- **Missing extra — install the SDK:**
  ```bash
  pip install attune-rag[claude]   # for ClaudeProvider
  pip install attune-rag[gemini]   # for GeminiProvider
  ```

- **Wrong provider name passed to `get_provider()`:**
  The `ValueError` message includes the list of known providers. Use that list, or call `list_available()` first:
  ```python
  from attune_rag.providers import get_provider, list_available
  print(list_available())          # e.g. ['claude', 'gemini']
  provider = get_provider("claude", api_key="YOUR_KEY")
  ```

- **API key not reaching the provider:**
  Pass it explicitly rather than relying on environment variable propagation:
  ```python
  provider = ClaudeProvider(api_key="sk-ant-...")
  ```
  Alternatively, export the variable in the same shell session that runs your code:
  ```bash
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```

- **SDK version mismatch:**
  A Claude or Gemini SDK upgrade can silently change model names, response shapes, or authentication flows. Pin and verify versions:
  ```bash
  pip show anthropic google-generativeai
  ```
  If you recently upgraded, check the SDK's changelog for breaking changes and roll back if necessary:
  ```bash
  pip install anthropic==<last-known-good>
  ```

- **Calling `generate_with_citations()` on `GeminiProvider`:**
  This method is only implemented on `ClaudeProvider`. Switch providers, or use `generate()` if citation tracking is not required.

## Source files

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/gemini.py`

**Tags:** `providers`, `llm`, `claude`, `gemini`
