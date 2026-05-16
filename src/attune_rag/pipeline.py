"""Pipeline orchestration: corpus + retriever + provenance + prompt."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from .prompts import (
    build_augmented_prompt,
    join_context,
    join_context_numbered,
)
from .provenance import CitationRecord, ClaimCitation, build_citation_record
from .retrieval import KeywordRetriever, RetrieverProtocol

if TYPE_CHECKING:
    from .corpus.base import CorpusProtocol
    from .expander import QueryExpander
    from .providers.base import LLMProvider
    from .reranker import LLMReranker

logger = structlog.get_logger(__name__)


#: Splitter inserted before the per-call user request block in
#: augmented prompts. Everything before this marker is stable
#: across calls (system rules + retrieved passages) and is the
#: candidate for Anthropic prompt caching.
_CACHE_SPLIT = "\n### USER REQUEST\n"

#: Minimum prefix length (chars) before we bother flagging the
#: prompt for caching. Anthropic only caches blocks of at least
#: ~1024 tokens; below that the cache_control marker is wasted.
_MIN_CACHE_CHARS = 1024


FALLBACK_PROMPT_TEMPLATE = """### USER REQUEST

{query}

### INSTRUCTION

No grounding context was found in the corpus for this
request. Answer honestly about what you do and do not
know. Do not invent attune APIs, workflow names, or CLI
commands. If the user is asking about something outside
the corpus's scope, say so."""


@dataclass(frozen=True)
class RagResult:
    """Output of ``RagPipeline.run``.

    ``context`` is the rendered passage block that was
    inserted into the prompt, preserved so evaluators
    (e.g. faithfulness judges) can score answers against
    the exact same text the generator saw. Empty string
    when ``fallback_used`` is True.

    ``claim_citations`` is populated only on the native-citations
    code path (when ``run_and_generate`` is called with
    ``use_native_citations=True`` and the provider supports it).
    On the legacy ``[P{n}]``-marker path it remains an empty
    tuple. ``used_native_citations`` records which path actually
    ran so callers can render output appropriately even when the
    requested path was unavailable (e.g. provider fallback).
    """

    augmented_prompt: str
    citation: CitationRecord
    confidence: float
    fallback_used: bool
    elapsed_ms: float
    context: str = ""
    claim_citations: tuple[ClaimCitation, ...] = ()
    used_native_citations: bool = False


class RagPipeline:
    """LLM-agnostic RAG pipeline.

    Given a query, runs retrieval against a
    ``CorpusProtocol`` and returns an augmented prompt plus
    provenance. Does NOT call an LLM; that's the consumer's
    choice (see ``run_and_generate`` for an optional
    convenience — task 1.8).
    """

    def __init__(
        self,
        corpus: CorpusProtocol | None = None,
        retriever: RetrieverProtocol | None = None,
        expander: QueryExpander | None = None,
        reranker: LLMReranker | None = None,
    ) -> None:
        """Wire up retrieval, optional query expansion, and optional reranking.

        Args:
            corpus: Source of retrievable entries. If ``None``,
                the pipeline lazily falls back to
                :class:`AttuneHelpCorpus` on first ``corpus``
                access — see :meth:`_default_corpus`.
            retriever: Strategy for scoring/ranking entries.
                Defaults to :class:`KeywordRetriever` when
                ``None``.
            expander: Optional :class:`QueryExpander`. When set,
                :meth:`_retrieve` synonym-expands the query
                before passing it to the retriever (joined as
                ``"original expansion1 expansion2 …"`` so the
                original terms still dominate scoring).
            reranker: Optional :class:`LLMReranker`. When set,
                :meth:`_retrieve` over-fetches by the reranker's
                ``candidate_multiplier`` and re-orders the
                widened candidate pool with an LLM call before
                trimming to ``k``.
        """
        self._corpus = corpus
        self.retriever = retriever or KeywordRetriever()
        self.expander = expander
        self.reranker = reranker

    @property
    def corpus(self) -> CorpusProtocol:
        """The corpus this pipeline retrieves from.

        Lazy: if no ``corpus`` was passed to ``__init__``, the
        first access triggers :meth:`_default_corpus` and the
        result is cached on the instance. Subsequent accesses
        return the cached corpus.
        """
        if self._corpus is None:
            self._corpus = self._default_corpus()
        return self._corpus

    @staticmethod
    def _default_corpus() -> CorpusProtocol:
        """Fall back to :class:`AttuneHelpCorpus` when no corpus is provided.

        Raises ``RuntimeError`` (chained from ``ImportError``)
        with an actionable message if the ``[attune-help]``
        extra is not installed, so the failure mode is "tell me
        what to install" instead of "module not found."
        """
        try:
            from .corpus.attune_help import AttuneHelpCorpus
        except ImportError as exc:
            raise RuntimeError(
                "No corpus provided and AttuneHelpCorpus is unavailable. "
                "Either pass a corpus= (e.g. DirectoryCorpus) or install "
                "'attune-rag[attune-help]'."
            ) from exc
        return AttuneHelpCorpus.from_attune_help()

    def _retrieve(self, query: str, k: int) -> list:
        """Run retrieval (with optional expansion + reranking).

        Extracted from :meth:`run` so :meth:`run_and_generate`
        can access the actual hit objects (with full content) on
        the native-citations path. Pure helper — no side effects.
        """
        retrieval_query = query
        if self.expander is not None:
            expansions = self.expander.expand(query)
            if expansions:
                retrieval_query = " ".join([query, *expansions])
        retrieval_k = k * self.reranker.candidate_multiplier if self.reranker else k
        hits = list(self.retriever.retrieve(retrieval_query, self.corpus, k=retrieval_k))
        if self.reranker is not None and hits:
            hits = self.reranker.rerank(query, hits)[:k]
        return hits

    def run(
        self,
        query: str,
        k: int = 3,
        prompt_variant: str = "citation",
    ) -> RagResult:
        """Retrieve, assemble, and return an augmented prompt + citation.

        Args:
            query: The user's question.
            k: Top-k hits to retrieve.
            prompt_variant: Which prompt template to use.
                One of ``baseline``, ``strict``, ``citation``,
                ``anti_prior``. See :mod:`attune_rag.prompts`.

                Defaults to ``citation``. Selected via A/B
                sweep on 2026-04-19: hallucination rate
                dropped 46.7% → 6.7% vs ``baseline`` while
                keeping retrieval quality identical. See
                ``docs/rag/faithfulness-decision-2026-04-19.md``.
        """
        start = time.perf_counter()
        now = datetime.now(timezone.utc)

        hits = self._retrieve(query, k)

        citation = build_citation_record(
            query=query,
            hits=hits,
            retriever_name=type(self.retriever).__name__,
            retrieved_at=now,
        )

        if not hits:
            augmented_prompt = FALLBACK_PROMPT_TEMPLATE.format(query=query.strip())
            context = ""
            confidence = 0.0
            fallback_used = True
        else:
            if prompt_variant == "citation":
                context = join_context_numbered(hits)
            else:
                context = join_context(hits)
            augmented_prompt = build_augmented_prompt(query, context, variant=prompt_variant)
            top_score = hits[0].score
            min_score = getattr(self.retriever, "MIN_SCORE", 1.0) or 1.0
            confidence = min(top_score / (min_score * 2), 1.0)
            fallback_used = False

        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "rag.run",
            query=query,
            hit_count=len(hits),
            fallback_used=fallback_used,
            confidence=round(confidence, 3),
            elapsed_ms=round(elapsed_ms, 2),
            corpus=self.corpus.name,
            retriever=type(self.retriever).__name__,
            prompt_variant=prompt_variant,
        )

        return RagResult(
            augmented_prompt=augmented_prompt,
            citation=citation,
            confidence=confidence,
            fallback_used=fallback_used,
            elapsed_ms=elapsed_ms,
            context=context,
        )

    async def run_and_generate(
        self,
        query: str,
        provider: LLMProvider | str,
        k: int = 3,
        model: str | None = None,
        max_tokens: int = 2048,
        prompt_variant: str = "citation",
        use_native_citations: bool = False,
    ) -> tuple[str, RagResult]:
        """Retrieve, build the augmented prompt, and call an LLM.

        ``provider`` may be an ``LLMProvider`` instance or a name
        string (``"claude"``, ``"gemini"``). In the string case,
        the matching provider is constructed with default
        credentials (from env vars).

        ``prompt_variant`` selects the prompt template. See
        :mod:`attune_rag.prompts`.

        ``use_native_citations`` opts into the Anthropic Citations
        API path: each retrieved hit becomes a ``custom_content``
        document block, and the model emits claim-level citations
        attached to its response text. The returned ``RagResult``
        carries ``claim_citations`` (and ``used_native_citations``)
        so callers can render with
        :func:`provenance.format_claim_citations_markdown`.

        Behavior matrix:

        - ``use_native_citations=False`` → existing prompt-assembly
          path. Returned ``RagResult.claim_citations == ()``.
        - ``use_native_citations=True`` and provider supports it
          and hits exist → native citations path.
        - ``use_native_citations=True`` and provider supports it
          but hits are empty → fallback prompt (no docs to cite);
          ``used_native_citations=False`` because the citations
          API was never called.
        - ``use_native_citations=True`` and provider does NOT
          support native citations → log warning, run existing
          prompt-assembly path; ``used_native_citations=False``.

        Returns ``(response_text, rag_result)``.
        """
        if isinstance(provider, str):
            from .providers import get_provider

            provider = get_provider(provider)

        if use_native_citations and not getattr(provider, "supports_native_citations", False):
            logger.warning(
                "rag.native_citations_unsupported",
                provider=getattr(provider, "name", type(provider).__name__),
                fallback="prompt_assembly",
            )
            use_native_citations = False

        if not use_native_citations:
            rag_result = self.run(query, k=k, prompt_variant=prompt_variant)
            prompt = rag_result.augmented_prompt
            cached_prefix: str | None = None
            split_idx = prompt.find(_CACHE_SPLIT)
            if split_idx != -1 and split_idx >= _MIN_CACHE_CHARS:
                cached_prefix = prompt[: split_idx + len(_CACHE_SPLIT)]
            response = await provider.generate(
                prompt,
                model=model,
                max_tokens=max_tokens,
                cached_prefix=cached_prefix,
            )
            return response, rag_result

        # Native citations path. We need the actual hit objects (full
        # content) to build the document payload, so we run retrieval
        # here instead of calling self.run().
        return await self._run_native_citations(
            query=query,
            provider=provider,
            k=k,
            model=model,
            max_tokens=max_tokens,
            prompt_variant=prompt_variant,
        )

    async def _run_native_citations(
        self,
        query: str,
        provider: LLMProvider,
        k: int,
        model: str | None,
        max_tokens: int,
        prompt_variant: str,
    ) -> tuple[str, RagResult]:
        """Native citations path. Internal — see
        :meth:`run_and_generate` for the public entry point.

        Retrieval is run identically to :meth:`run`. When hits
        exist we send them as ``CitationDocument``s through the
        provider's citations API; the response is parsed back
        into ``claim_citations`` on the returned ``RagResult``.
        When hits are empty we delegate to :meth:`run` for the
        fallback prompt and a plain ``provider.generate`` call —
        ``used_native_citations`` stays ``False`` because the
        citations API was never engaged.
        """
        from .providers.base import CitationDocument

        start = time.perf_counter()
        now = datetime.now(timezone.utc)
        hits = self._retrieve(query, k)
        citation = build_citation_record(
            query=query,
            hits=hits,
            retriever_name=type(self.retriever).__name__,
            retrieved_at=now,
        )

        if not hits:
            # No documents to cite — fall back to the legacy fallback
            # prompt path. Note: we still report use_native_citations=False
            # because no citations call was made.
            augmented_prompt = FALLBACK_PROMPT_TEMPLATE.format(query=query.strip())
            response = await provider.generate(
                augmented_prompt,
                model=model,
                max_tokens=max_tokens,
                cached_prefix=None,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info(
                "rag.run",
                query=query,
                hit_count=0,
                fallback_used=True,
                confidence=0.0,
                elapsed_ms=round(elapsed_ms, 2),
                corpus=self.corpus.name,
                retriever=type(self.retriever).__name__,
                prompt_variant=prompt_variant,
                native_citations_requested=True,
                native_citations_used=False,
            )
            rag_result = RagResult(
                augmented_prompt=augmented_prompt,
                citation=citation,
                confidence=0.0,
                fallback_used=True,
                elapsed_ms=elapsed_ms,
                context="",
                claim_citations=(),
                used_native_citations=False,
            )
            return response, rag_result

        documents = [CitationDocument(title=h.entry.path, text=h.entry.content) for h in hits]
        # Render context too, for parity with the legacy path so callers
        # that inspect rag_result.context for evals see the same input.
        if prompt_variant == "citation":
            context = join_context_numbered(hits)
        else:
            context = join_context(hits)

        cited = await provider.generate_with_citations(
            documents=documents,
            query=query,
            model=model,
            max_tokens=max_tokens,
        )

        top_score = hits[0].score
        min_score = getattr(self.retriever, "MIN_SCORE", 1.0) or 1.0
        confidence = min(top_score / (min_score * 2), 1.0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "rag.run",
            query=query,
            hit_count=len(hits),
            fallback_used=False,
            confidence=round(confidence, 3),
            elapsed_ms=round(elapsed_ms, 2),
            corpus=self.corpus.name,
            retriever=type(self.retriever).__name__,
            prompt_variant=prompt_variant,
            native_citations_requested=True,
            native_citations_used=True,
            claim_citation_count=len(cited.claim_citations),
        )

        # augmented_prompt is empty on this path: the model didn't see a
        # rendered numbered-passage prompt; documents went over the wire
        # as structured blocks. We surface this as an empty string rather
        # than fabricating a prompt that wasn't actually sent.
        rag_result = RagResult(
            augmented_prompt="",
            citation=citation,
            confidence=confidence,
            fallback_used=False,
            elapsed_ms=elapsed_ms,
            context=context,
            claim_citations=cited.claim_citations,
            used_native_citations=True,
        )
        return cited.text, rag_result
