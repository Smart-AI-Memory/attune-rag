"""Pipeline orchestration: corpus + retriever + provenance + prompt."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from .prompts import build_augmented_prompt, join_context
from .provenance import CitationRecord, build_citation_record
from .retrieval import KeywordRetriever, RetrieverProtocol

if TYPE_CHECKING:
    from .corpus.base import CorpusProtocol

logger = structlog.get_logger(__name__)


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
    """Output of ``RagPipeline.run``."""

    augmented_prompt: str
    citation: CitationRecord
    confidence: float
    fallback_used: bool
    elapsed_ms: float


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
    ) -> None:
        self._corpus = corpus
        self.retriever = retriever or KeywordRetriever()

    @property
    def corpus(self) -> CorpusProtocol:
        if self._corpus is None:
            self._corpus = self._default_corpus()
        return self._corpus

    @staticmethod
    def _default_corpus() -> CorpusProtocol:
        try:
            from .corpus.attune_help import AttuneHelpCorpus
        except ImportError as exc:
            raise RuntimeError(
                "No corpus provided and AttuneHelpCorpus is unavailable. "
                "Either pass a corpus= (e.g. DirectoryCorpus) or install "
                "'attune-rag[attune-help]'."
            ) from exc
        return AttuneHelpCorpus()

    def run(self, query: str, k: int = 3) -> RagResult:
        """Retrieve, assemble, and return an augmented prompt + citation."""
        start = time.perf_counter()
        now = datetime.now(timezone.utc)

        hits = list(self.retriever.retrieve(query, self.corpus, k=k))
        citation = build_citation_record(
            query=query,
            hits=hits,
            retriever_name=type(self.retriever).__name__,
            retrieved_at=now,
        )

        if not hits:
            augmented_prompt = FALLBACK_PROMPT_TEMPLATE.format(query=query.strip())
            confidence = 0.0
            fallback_used = True
        else:
            context = join_context(hits)
            augmented_prompt = build_augmented_prompt(query, context)
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
        )

        return RagResult(
            augmented_prompt=augmented_prompt,
            citation=citation,
            confidence=confidence,
            fallback_used=fallback_used,
            elapsed_ms=elapsed_ms,
        )

    async def run_and_generate(
        self,
        query: str,
        provider: LLMProvider | str,
        k: int = 3,
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> tuple[str, RagResult]:
        """Retrieve, build the augmented prompt, and call an LLM.

        ``provider`` may be an ``LLMProvider`` instance or a name
        string (``"claude"``, ``"openai"``, ``"gemini"``). In the
        string case, the matching provider is constructed with
        default credentials (from env vars).

        Returns ``(response_text, rag_result)``. Callers render
        citations with ``format_citations_markdown`` as needed.
        """
        if isinstance(provider, str):
            from .providers import get_provider

            provider = get_provider(provider)

        rag_result = self.run(query, k=k)
        response = await provider.generate(
            rag_result.augmented_prompt,
            model=model,
            max_tokens=max_tokens,
        )
        return response, rag_result
