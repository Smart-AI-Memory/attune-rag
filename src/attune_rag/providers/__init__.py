"""Optional LLM provider adapters.

Each adapter is behind a pip extra:

- attune-rag[claude]  -> ClaudeProvider
- attune-rag[openai]  -> OpenAIProvider
- attune-rag[gemini]  -> GeminiProvider

Adapters lazy-import their SDKs so attune-rag installs
cleanly without any provider SDK.
"""

from __future__ import annotations

from importlib import import_module

from .base import LLMProvider

_SDK_PROBES = {
    "claude": "anthropic",
    "openai": "openai",
    "gemini": "google.genai",
}


def list_available() -> list[str]:
    """Return the names of providers whose SDKs are importable.

    Does not instantiate clients or hit the network — just
    probes import availability. Callers use this for feature
    detection.
    """
    available: list[str] = []
    for name, module in _SDK_PROBES.items():
        try:
            import_module(module)
            available.append(name)
        except ImportError:
            continue
    return available


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Return an instance of the named provider.

    Raises ``ValueError`` for unknown names and ``RuntimeError``
    (from the adapter constructor) if the corresponding SDK is
    not installed.
    """
    if name == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(**kwargs)
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(**kwargs)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(**kwargs)
    raise ValueError(f"Unknown provider {name!r}. Known providers: {sorted(_SDK_PROBES)}.")


__all__ = ["LLMProvider", "list_available", "get_provider"]
