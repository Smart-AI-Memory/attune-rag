---
type: note
name: providers-note
feature: providers
depth: note
generated_at: 2026-05-20T03:27:22.854434+00:00
source_hash: fbe19e4accf7e90a0aec29d23dcdabe3822e7458ce7ff5e4412a81d42aae02f9
status: generated
---

# Note: providers

## Context

The `providers` module supplies optional async adapters that connect `attune-rag` to external LLM APIs. Each adapter is installed separately so that the core package has no mandatory SDK dependencies.

## Content

The module defines the `LLMProvider` protocol in `base.py`. Any conforming class must implement at least `generate()`; `generate_with_citations()` is available on providers that support document-grounded responses.

Two concrete implementations ship with the package:

- **`ClaudeProvider`** (`providers/claude.py`) — wraps Anthropic's async Messages API. Requires the `attune-rag[claude]` extra.
- **`GeminiProvider`** (`providers/gemini.py`) — wraps Google's async `genai` models API. Requires the `attune-rag[gemini]` extra.

Neither SDK is imported at module load time. The import happens inside the provider class, so installing only one extra does not affect the other.

Two helper functions are exported from `providers/__init__.py`:

- **`list_available()`** — returns the names of providers whose SDKs are currently importable. Use this to check which extras are installed at runtime.
- **`get_provider(name, **kwargs)`** — returns a ready-to-use `LLMProvider` instance for the given name. Raises `ValueError` if the name is unrecognized or the required SDK is not installed.

The citations-related dataclasses, `CitationDocument` and `CitedResponse`, are defined in `base.py` and used only with providers that implement `generate_with_citations()`. `GeminiProvider` does not currently implement that method; `ClaudeProvider` does.

> **Note:** The source files list a `providers/openai.py` module, but no `OpenAIProvider` class appears in the public API (`__all__`) or the installed extras. Treat OpenAI support as not yet available.

**Tags:** `providers`, `llm`, `claude`, `gemini`
