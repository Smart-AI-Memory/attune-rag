"""Pipeline orchestration. Implementation in task 1.6."""

from __future__ import annotations

from dataclasses import dataclass

from .provenance import CitationRecord
from .retrieval import KeywordRetriever, RetrieverProtocol


@dataclass(frozen=True)
class RagResult:
    """Result of RagPipeline.run.

    Populated fully in task 1.6. Task 1.1 defines the
    shape only.
    """

    augmented_prompt: str
    citation: CitationRecord
    confidence: float
    fallback_used: bool
    elapsed_ms: float


class RagPipeline:
    """LLM-agnostic RAG pipeline.

    Task 1.6 implements run(). Task 1.8 adds
    run_and_generate(). Task 1.1 scaffolds the class so
    imports resolve.
    """

    def __init__(
        self,
        corpus: object | None = None,
        retriever: RetrieverProtocol | None = None,
    ) -> None:
        self.corpus = corpus
        self.retriever = retriever or KeywordRetriever()

    def run(self, query: str, k: int = 3) -> RagResult:
        """Run retrieval + prompt assembly. Stub until task 1.6."""
        raise NotImplementedError(
            "RagPipeline.run is implemented in task 1.6 of " "the RAG grounding spec."
        )
