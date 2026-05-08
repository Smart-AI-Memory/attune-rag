"""LLMProvider protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..provenance import ClaimCitation


@dataclass(frozen=True)
class CitationDocument:
    """One source document passed to a citations-capable provider.

    For attune-rag, each retrieved hit becomes a single
    ``CitationDocument`` in the call to
    :meth:`LLMProvider.generate_with_citations`. ``title`` is
    surfaced back in returned :class:`ClaimCitation` records so
    callers can map citations directly to the originating hit.

    Attributes:
        title: Human-readable identifier for the document.
            attune-rag uses the template path
            (e.g. ``concepts/tool-security-audit.md``).
        text: The verbatim source content the model may cite from.
    """

    title: str
    text: str


@dataclass(frozen=True)
class CitedResponse:
    """Response from a citations-capable provider.

    ``text`` is the assembled response text (concatenation of all
    text blocks in the model's reply). ``claim_citations`` is the
    structured per-claim attribution emitted by the model — empty
    when the model declined to cite any claim.
    """

    text: str
    claim_citations: tuple[ClaimCitation, ...]


@runtime_checkable
class LLMProvider(Protocol):
    """An async LLM provider that consumes a prompt and returns text.

    Implementations live in ``attune_rag.providers.{claude,gemini}``
    behind optional extras. Each lazy-imports its SDK so core
    attune-rag installs cleanly without any provider deps.

    Providers that natively support claim-level citations (e.g. via
    Anthropic's Citations API) set
    :attr:`supports_native_citations` to ``True`` and implement
    :meth:`generate_with_citations`. The pipeline checks the flag
    before dispatching, so providers without native support never
    need to override the method.
    """

    name: str
    supports_native_citations: bool

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        cached_prefix: str | None = None,
    ) -> str:
        """Generate a response from ``prompt``.

        ``cached_prefix`` is an optional stable prefix the
        provider may flag for prompt caching when the
        underlying API supports it. Providers that do not
        support caching must accept the kwarg and ignore it
        — the pipeline always passes it through.
        """
        ...

    async def generate_with_citations(
        self,
        documents: list[CitationDocument],
        query: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> CitedResponse:
        """Generate a response with claim-level citations.

        The default implementation raises ``NotImplementedError``.
        Providers that cannot natively cite (e.g. Gemini today)
        leave this unimplemented and set
        :attr:`supports_native_citations` to ``False`` — the
        pipeline then falls back to the prompt-assembly path.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support native citations. "
            "Set supports_native_citations=False on the provider class "
            "and the pipeline will fall back to the prompt-assembly path."
        )
