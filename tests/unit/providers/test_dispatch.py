"""Unit tests for providers.list_available and get_provider."""

from __future__ import annotations

import pytest

from attune_rag.providers import get_provider, list_available


def test_list_available_returns_installed_providers() -> None:
    available = list_available()
    # In dev env all three extras install; assert the contract
    # rather than a specific list.
    assert isinstance(available, list)
    for name in available:
        assert name in {"claude", "openai", "gemini"}


def test_get_provider_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("mystery")


def test_get_provider_returns_instance_when_installed() -> None:
    for name in list_available():
        # Each constructor accepts api_key="fake" for smoke; we
        # don't call .generate here.
        provider = get_provider(name, api_key="fake")  # pragma: allowlist secret
        assert provider.name == name
