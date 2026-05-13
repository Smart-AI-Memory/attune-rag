"""Tests for attune_rag.providers.base — protocol shape + dataclass invariants.

Pins the LLMProvider protocol and the two dataclasses concrete providers
(claude, gemini) consume. These are part of attune-rag's contract surface
since downstream callers can implement their own provider by conforming
to the protocol.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import FrozenInstanceError

import pytest

from attune_rag.providers.base import (
    CitationDocument,
    CitedResponse,
    LLMProvider,
)

# ---------------------------------------------------------------------------
# CitationDocument
# ---------------------------------------------------------------------------


def test_citation_document_is_frozen() -> None:
    doc = CitationDocument(title="t.md", text="content")
    with pytest.raises(FrozenInstanceError):
        doc.title = "other.md"  # type: ignore[misc]


def test_citation_document_equality_by_field() -> None:
    a = CitationDocument(title="t.md", text="x")
    b = CitationDocument(title="t.md", text="x")
    assert a == b
    assert hash(a) == hash(b)


def test_citation_document_unequal_when_different_text() -> None:
    a = CitationDocument(title="t.md", text="one")
    b = CitationDocument(title="t.md", text="two")
    assert a != b


# ---------------------------------------------------------------------------
# CitedResponse
# ---------------------------------------------------------------------------


def test_cited_response_is_frozen() -> None:
    r = CitedResponse(text="ans", claim_citations=())
    with pytest.raises(FrozenInstanceError):
        r.text = "other"  # type: ignore[misc]


def test_cited_response_claim_citations_is_tuple() -> None:
    """``claim_citations`` is a tuple to keep the dataclass hashable."""
    r = CitedResponse(text="ans", claim_citations=())
    assert isinstance(r.claim_citations, tuple)


# ---------------------------------------------------------------------------
# LLMProvider protocol
# ---------------------------------------------------------------------------


def test_llm_provider_is_runtime_checkable() -> None:
    """Pipeline uses ``isinstance(p, LLMProvider)`` to dispatch — runtime check
    must work."""

    class _Conforming:
        name = "fake"
        supports_native_citations = False

        async def generate(
            self,
            prompt: str,
            model: str | None = None,
            max_tokens: int = 2048,
            cached_prefix: str | None = None,
        ) -> str:
            return "ok"

        async def generate_with_citations(self, *args, **kwargs):  # noqa: ANN
            raise NotImplementedError

    assert isinstance(_Conforming(), LLMProvider)


def test_llm_provider_rejects_missing_attrs() -> None:
    """Class missing ``name`` or required methods must NOT pass isinstance."""

    class _Incomplete:
        # Missing name + supports_native_citations + methods.
        pass

    assert not isinstance(_Incomplete(), LLMProvider)


def test_default_generate_with_citations_raises_not_implemented() -> None:
    """Concrete providers that don't override the citations method must
    surface a clear NotImplementedError when callers reach for it."""

    class _NoCitations:
        name = "nocite"
        supports_native_citations = False

        async def generate(
            self,
            prompt: str,
            model: str | None = None,
            max_tokens: int = 2048,
            cached_prefix: str | None = None,
        ) -> str:
            return "ok"

        # Inherit default behavior — call via the protocol method directly.
        generate_with_citations = LLMProvider.generate_with_citations

    inst = _NoCitations()
    with pytest.raises(NotImplementedError, match="does not support native citations"):
        asyncio.run(inst.generate_with_citations(documents=[], query="q"))


# ---------------------------------------------------------------------------
# Signature pins — providers depend on these defaults
# ---------------------------------------------------------------------------


def test_generate_signature_keeps_documented_kwargs() -> None:
    sig = inspect.signature(LLMProvider.generate)
    params = sig.parameters
    # Pipeline + tests pass these by keyword.
    for required in ("prompt", "model", "max_tokens", "cached_prefix"):
        assert required in params, f"generate() lost ``{required}`` kwarg"
    assert params["max_tokens"].default == 2048


def test_generate_with_citations_signature_keeps_documented_kwargs() -> None:
    sig = inspect.signature(LLMProvider.generate_with_citations)
    params = sig.parameters
    for required in ("documents", "query", "system", "model", "max_tokens"):
        assert required in params, f"generate_with_citations() lost ``{required}`` kwarg"
    assert params["max_tokens"].default == 2048
