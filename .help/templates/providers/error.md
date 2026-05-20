---
type: error
name: providers-error
feature: providers
depth: error
generated_at: 2026-05-20T03:27:22.836828+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Providers errors

## Common error signatures

Errors in the `providers` module fall into two categories: missing optional SDK dependencies and invalid provider names passed to `get_provider()`.

| Exception | Message pattern | Likely cause |
|---|---|---|
| `ValueError` | `Unknown provider '{name}'. Known providers: {list}.` | You passed an unrecognized name to `get_provider()`. |
| `ImportError` / `ModuleNotFoundError` | *(varies by SDK)* | You instantiated `ClaudeProvider` or `GeminiProvider` without installing the required extra. |

## Where errors originate

- **`get_provider(name, **kwargs)`** — Raises `ValueError` if `name` does not match any registered provider. The error message includes both the unknown name and the list of recognized provider names.
- **`ClaudeProvider.__init__()`** — Requires the `attune-rag[claude]` extra. If the `anthropic` SDK is not installed, constructing this class raises an `ImportError`.
- **`GeminiProvider.__init__()`** — Requires the `attune-rag[gemini]` extra. If the `google-genai` SDK is not installed, constructing this class raises an `ImportError`.
- **`list_available()`** — Returns only the providers whose SDKs are importable at call time. If this list is shorter than expected, a required extra is missing rather than an exception being raised.

## How to diagnose

1. **Check the provider name.** If you see `ValueError: Unknown provider '...'. Known providers: ...`, the name you passed to `get_provider()` is misspelled or not registered. Call `list_available()` to see which providers are importable in the current environment.

2. **Verify the installed extras.** If you get an `ImportError` when constructing `ClaudeProvider` or `GeminiProvider`, install the corresponding extra:
   - `ClaudeProvider` → `pip install attune-rag[claude]`
   - `GeminiProvider` → `pip install attune-rag[gemini]`

3. **Confirm `list_available()` output matches your expectations.** If a provider you expect is absent from the returned list, its SDK is not importable — install the relevant extra and re-check.

4. **Check API key availability.** Both `ClaudeProvider` and `GeminiProvider` accept an `api_key` argument. If you rely on environment-variable injection instead, confirm the variable is set before the provider is constructed.

## Source files

- `src/attune_rag/providers/__init__.py`
- `src/attune_rag/providers/base.py`
- `src/attune_rag/providers/claude.py`
- `src/attune_rag/providers/gemini.py`

**Tags:** `providers`, `llm`, `claude`, `gemini`
