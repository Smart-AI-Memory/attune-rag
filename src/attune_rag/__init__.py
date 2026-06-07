"""Lightweight, LLM-agnostic RAG pipeline.

Core public API:

- RagPipeline — orchestrates retrieval + prompt assembly
- RagResult — return value of RagPipeline.run
- CitationRecord, CitedSource — provenance records
- CorpusProtocol, RetrievalEntry — pluggable corpus interface
- DirectoryCorpus — generic markdown directory loader
- AttuneHelpCorpus — attune-help bundled corpus (opt. dep)
- KeywordRetriever — default retriever (no embedding)
- TransformerRetriever — opt-in dense retriever ([transformers] extra)
- build_augmented_prompt — prompt template helper

See spec: attune-ai/.claude/plans/feature-rag-code-
grounding-2026-04-17.md (v4.0).

Subsequent tasks in the spec fill in each module.
"""

from __future__ import annotations

__version__ = "0.2.0"

# NOTE: Imports are added incrementally as tasks 1.2-1.8
# land. For task 1.1 (scaffold only) the public names
# below resolve to minimal stubs so imports work for CI.
from .corpus import (
    AttuneHelpCorpus,
    CorpusProtocol,
    DirectoryCorpus,
    RetrievalEntry,
)
from .embedding import EmbeddingRetriever
from .expander import QueryExpander
from .hybrid import HybridRetriever
from .pipeline import RagPipeline, RagResult
from .prompts import PROMPT_VARIANTS, build_augmented_prompt
from .provenance import (
    CitationRecord,
    CitedSource,
    ClaimCitation,
    format_citations_markdown,
    format_claim_citations_markdown,
)
from .reranker import LLMReranker
from .retrieval import KeywordRetriever, RetrievalHit, RetrieverProtocol
from .transformer import TransformerRetriever

__all__ = [
    "RagPipeline",
    "RagResult",
    "CitationRecord",
    "CitedSource",
    "ClaimCitation",
    "format_citations_markdown",
    "format_claim_citations_markdown",
    "CorpusProtocol",
    "RetrievalEntry",
    "DirectoryCorpus",
    "AttuneHelpCorpus",
    "KeywordRetriever",
    "EmbeddingRetriever",
    "HybridRetriever",
    "TransformerRetriever",
    "RetrievalHit",
    "RetrieverProtocol",
    "build_augmented_prompt",
    "PROMPT_VARIANTS",
    "QueryExpander",
    "LLMReranker",
]
