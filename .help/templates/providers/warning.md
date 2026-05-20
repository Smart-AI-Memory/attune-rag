---
type: warning
name: providers-warning
feature: providers
depth: warning
generated_at: 2026-05-20T03:27:22.842192+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Providers cautions

## What to watch for

The `providers` module ships optional LLM adapters for Claude and Gemini behind the `LLMProvider` async protocol. Because each adapter depends on an optional SDK extra (`attune-rag[claude]` or `attune-rag[gemini]`), availability is determined at runtime rather than at install time. This means a provider that works in one environment may silently be absent in another.

## Risk areas

**`get_provider()` raises on unknown or uninstalled providers.**
If you request a provider whose SDK is not installed, `get_provider()` raises `ValueError` with the message `'Unknown provider {...}. Known providers: {...}.'`. The "known providers" list reflects only what is currently importable — not every provider that exists in the codebase. Call `list_available()` first and check that your target provider is in the returned list before calling `get_provider()`.

**`list_available()` reflects the runtime environment, not your intent.**
`list_available()` returns only the providers whose SDKs are importable in the current process. In a minimal install or a CI environment without the optional extras, the list may be empty or missing providers you expect. Do not use the result of `list_available()` as a static constant — re-query it if the environment may have changed between calls.

**`GeminiProvider` does not implement `generate_with_citations()`.**
`ClaudeProvider` implements both `generate()` and `generate_with_citations()`, but `GeminiProvider` only implements `generate()`. If your code depends on `generate_with_citations()` and you swap providers, you will get an `AttributeError` at call time. Check which methods your code actually uses before switching between providers.

**`cached_prefix` in `generate()` is silently ignored by providers that do not support it.**
The `cached_prefix` parameter on `generate()` is part of the `LLMProvider` protocol signature, but not every concrete provider acts on it. Passing a `cached_prefix` to a provider that ignores it will not raise an error — the prefix will simply be omitted, which can change output quality or cost unexpectedly.

**API keys fall back to environment variables.**
Both `ClaudeProvider` and `GeminiProvider` accept `api_key=None`, in which case they read credentials from the environment. A missing or stale environment variable will cause authentication failures at call time, not at construction time. Verify your credentials are set correctly before making the first `generate()` call.

## How to avoid problems

1. **Guard provider access with `list_available()`.**
   Before calling `get_provider()`, confirm the provider is present:
   ```python
   if "claude" not in list_available():
       raise EnvironmentError("Install attune-rag[claude] to use ClaudeProvider.")
   provider = get_provider("claude", api_key=MY_KEY)
   ```

2. **Install the correct extras for your target providers.**
   Use `attune-rag[claude]` for `ClaudeProvider` and `attune-rag[gemini]` for `GeminiProvider`. Without the corresponding extra, the provider will not appear in `list_available()` and `get_provider()` will raise.

3. **Check method support before swapping providers.**
   If your pipeline calls `generate_with_citations()`, confirm that the provider you are switching to exposes that method. Currently only `ClaudeProvider` does.

4. **Validate credentials at startup.**
   Make a lightweight test call during application startup to confirm that credentials are valid. This surfaces authentication errors early rather than mid-request.

5. **Inject clients in tests instead of relying on live credentials.**
   Both `ClaudeProvider` and `GeminiProvider` accept a pre-built `client` argument (`AsyncAnthropic` or `GenAIClient`). Pass a mock client in tests to avoid hitting real APIs and to keep tests deterministic across environments.

## Source files

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/gemini.py`

**Tags:** `providers`, `llm`, `claude`, `gemini`
