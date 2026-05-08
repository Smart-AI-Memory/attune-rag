"""ClaudeProvider — requires ``attune-rag[claude]`` extra."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..provenance import ClaimCitation
from .base import CitationDocument, CitedResponse

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


# Per-request document ceiling enforced by attune-rag.
# The Anthropic Citations API itself accepts well above this — the V3
# probe (2026-05-08) confirmed n=200 documents accepted without
# rejection, with the real cap higher still. We pin 200 here as a
# practical ceiling: it covers the 20–50 docs an attune-rag retrieval
# realistically sends, leaves headroom for future k bumps, and
# surfaces a clean ValueError instead of an opaque 400 if a caller
# tries to send hundreds.
MAX_CITATION_DOCUMENTS = 200


class ClaudeProvider:
    """Thin async wrapper over Anthropic's Messages API.

    Lazy-imports ``anthropic`` so attune-rag installs cleanly
    without the Claude SDK. Supports the native Citations API
    via :meth:`generate_with_citations`; falls back to plain
    text generation via :meth:`generate`.
    """

    name = "claude"
    DEFAULT_MODEL = "claude-sonnet-4-6"
    supports_native_citations = True

    def __init__(
        self,
        api_key: str | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "ClaudeProvider requires the [claude] extra. "
                "Install with: pip install 'attune-rag[claude]'"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        max_tokens: int = 2048,
        cached_prefix: str | None = None,
    ) -> str:
        if cached_prefix:
            # Two-block message: stable context (cached) + dynamic tail
            tail = prompt[len(cached_prefix) :]
            content = [
                {
                    "type": "text",
                    "text": cached_prefix,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": tail},
            ]
        else:
            content = prompt

        response = await self._client.messages.create(
            model=model or self.DEFAULT_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        chunks: list[str] = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "".join(chunks)

    async def generate_with_citations(
        self,
        documents: list[CitationDocument],
        query: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> CitedResponse:
        """Generate a response with claim-level citations.

        Each ``CitationDocument`` is sent as a separate
        ``custom_content`` document block with ``citations.enabled``
        set, so returned ``document_index`` values map directly
        back into the ``documents`` list (one-doc-per-hit layout).

        Empty ``documents`` is rejected — callers should fall back
        to :meth:`generate` for the no-context path.
        """
        if not documents:
            raise ValueError(
                "generate_with_citations requires at least one document; "
                "for the no-context path call generate() instead."
            )
        if len(documents) > MAX_CITATION_DOCUMENTS:
            raise ValueError(
                f"generate_with_citations accepts at most "
                f"{MAX_CITATION_DOCUMENTS} documents; got {len(documents)}. "
                "Reduce k or batch the calls."
            )

        content: list[dict[str, Any]] = self._build_documents_payload(documents)
        content.append({"type": "text", "text": query})

        kwargs: dict[str, Any] = {
            "model": model or self.DEFAULT_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system is not None:
            kwargs["system"] = system

        response = await self._client.messages.create(**kwargs)
        return self._parse_cited_response(response)

    @staticmethod
    def _build_documents_payload(
        documents: list[CitationDocument],
    ) -> list[dict[str, Any]]:
        """Render documents as ``custom_content`` document blocks.

        One block per document keeps ``document_index`` aligned
        with the input list.

        ``cache_control: ephemeral`` is attached to the **first**
        document so the entire document prefix is cached together.
        Empirically verified (V2 probe, 2026-05-08): a 3799-token
        document payload yielded full cache hits on the second
        call (``cache_read_input_tokens=3799``,
        ``cache_creation_input_tokens=0``) with ~30% latency
        improvement on the cached call. Document-block caching
        behaves identically to text-block caching for our
        purposes; the marker on the first document covers all
        subsequent documents in the same request.
        """
        payload: list[dict[str, Any]] = []
        for i, doc in enumerate(documents):
            block: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "content",
                    "content": [{"type": "text", "text": doc.text}],
                },
                "title": doc.title,
                "citations": {"enabled": True},
            }
            if i == 0:
                block["cache_control"] = {"type": "ephemeral"}
            payload.append(block)
        return payload

    @staticmethod
    def _parse_cited_response(response: Any) -> CitedResponse:
        """Convert an Anthropic Messages response to ``CitedResponse``.

        Walks ``response.content`` collecting text blocks and
        flattening their per-claim citations. Char offsets in
        ``response_span`` are computed from the assembled text
        so consumers can highlight or footnote the claim in the
        rendered response without re-walking the raw blocks.

        Citation field-name handling: Anthropic returns one of
        several citation location types — for ``custom_content``
        sources we get ``CitationContentBlockLocation`` with
        ``start_block_index`` / ``end_block_index``. We capture
        ``start_block_index`` (always 0 with our one-block-per-doc
        layout). The field is read defensively via getattr so
        char- and page-location citations don't blow up here.
        """
        text_chunks: list[str] = []
        claim_citations: list[ClaimCitation] = []
        running_offset = 0

        for block in response.content:
            if getattr(block, "type", None) != "text":
                continue
            text = getattr(block, "text", "") or ""
            text_chunks.append(text)
            span_start = running_offset
            span_end = running_offset + len(text)
            running_offset = span_end

            citations = getattr(block, "citations", None) or []
            for cite in citations:
                claim_citations.append(
                    ClaimCitation(
                        response_span=(span_start, span_end),
                        document_index=getattr(cite, "document_index", 0) or 0,
                        document_title=getattr(cite, "document_title", "") or "",
                        cited_text=getattr(cite, "cited_text", "") or "",
                        cited_block_index=getattr(cite, "start_block_index", 0) or 0,
                    )
                )

        return CitedResponse(
            text="".join(text_chunks),
            claim_citations=tuple(claim_citations),
        )
