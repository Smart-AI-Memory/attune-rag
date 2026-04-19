"""Lightweight, LLM-agnostic RAG pipeline.

Core public API:

- RagPipeline — orchestrates retrieval + prompt assembly
- RagResult — return value of RagPipeline.run
- CitationRecord, CitedSource — provenance records
- CorpusProtocol, RetrievalEntry — pluggable corpus interface
- DirectoryCorpus — generic markdown directory loader
- AttuneHelpCorpus — attune-help bundled corpus (opt. dep)
- KeywordRetriever — default retriever (no embedding)
- build_augmented_prompt — prompt template helper

See spec: attune-ai/.claude/plans/feature-rag-code-
grounding-2026-04-17.md (v4.0).

Subsequent tasks in the spec fill in each module.
"""

__version__ = "0.1.3"

# NOTE: Imports are added incrementally as tasks 1.2-1.8
# land. For task 1.1 (scaffold only) the public names
# below resolve to minimal stubs so imports work for CI.
from .corpus import CorpusProtocol, DirectoryCorpus, RetrievalEntry
from .pipeline import RagPipeline, RagResult
from .prompts import PROMPT_VARIANTS, build_augmented_prompt
from .provenance import CitationRecord, CitedSource, format_citations_markdown
from .retrieval import KeywordRetriever, RetrievalHit, RetrieverProtocol

__all__ = [
    "RagPipeline",
    "RagResult",
    "CitationRecord",
    "CitedSource",
    "format_citations_markdown",
    "CorpusProtocol",
    "RetrievalEntry",
    "DirectoryCorpus",
    "KeywordRetriever",
    "RetrievalHit",
    "RetrieverProtocol",
    "build_augmented_prompt",
    "PROMPT_VARIANTS",
]
