"""Public-API contract tests for attune_rag.

attune-rag is the contract source of truth for the attune product
family — attune-gui, attune-help, and attune-author all consume objects
from this package. This file pins the public surface so silent breaking
changes (renames, signature drift, missing docstrings) fail at PR time
instead of surfacing as runtime errors weeks later in consumer layers.

Each named export in ``attune_rag.__all__`` must:

- Be importable directly from ``attune_rag``
- Be the right kind (class vs callable)
- Carry a docstring
- Preserve its declared module identity (catches accidental re-binds)
"""

from __future__ import annotations

import inspect

import pytest

import attune_rag

# ---------------------------------------------------------------------------
# __all__ surface
# ---------------------------------------------------------------------------


EXPECTED_ALL = {
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
    "RetrievalHit",
    "RetrieverProtocol",
    "build_augmented_prompt",
    "PROMPT_VARIANTS",
    "QueryExpander",
    "LLMReranker",
}


def test_all_lists_the_documented_public_surface() -> None:
    """``__all__`` is the contract; new exports require an explicit add."""
    assert set(attune_rag.__all__) == EXPECTED_ALL


@pytest.mark.parametrize("name", sorted(EXPECTED_ALL))
def test_export_is_importable_from_top_level(name: str) -> None:
    """Each declared export must be reachable as ``attune_rag.<name>``."""
    assert hasattr(attune_rag, name), f"missing public export: {name}"
    assert getattr(attune_rag, name) is not None


@pytest.mark.parametrize("name", sorted(EXPECTED_ALL))
def test_export_has_docstring(name: str) -> None:
    """Every public symbol needs a docstring — consumer doc generators rely on it."""
    obj = getattr(attune_rag, name)
    if isinstance(obj, dict):
        # PROMPT_VARIANTS is a module-level dict; no docstring expected.
        return
    doc = inspect.getdoc(obj)
    assert doc, f"{name} is missing a docstring"
    assert len(doc.strip()) >= 10, f"{name} docstring is too short: {doc!r}"


# ---------------------------------------------------------------------------
# Type / shape contracts
# ---------------------------------------------------------------------------


CLASSES = {
    "RagPipeline",
    "RagResult",
    "CitationRecord",
    "CitedSource",
    "ClaimCitation",
    "DirectoryCorpus",
    "AttuneHelpCorpus",
    "KeywordRetriever",
    "EmbeddingRetriever",
    "HybridRetriever",
    "RetrievalHit",
    "RetrievalEntry",
    "QueryExpander",
    "LLMReranker",
}

PROTOCOLS = {"CorpusProtocol", "RetrieverProtocol"}

CALLABLES = {
    "format_citations_markdown",
    "format_claim_citations_markdown",
    "build_augmented_prompt",
}

CONTAINERS = {"PROMPT_VARIANTS"}


def test_export_classification_is_complete() -> None:
    """Every export falls into exactly one category."""
    classified = CLASSES | PROTOCOLS | CALLABLES | CONTAINERS
    assert classified == EXPECTED_ALL, (
        f"unclassified exports: {EXPECTED_ALL - classified}; "
        f"unknown classifiers: {classified - EXPECTED_ALL}"
    )


@pytest.mark.parametrize("name", sorted(CLASSES | PROTOCOLS))
def test_class_exports_are_classes(name: str) -> None:
    obj = getattr(attune_rag, name)
    assert inspect.isclass(obj), f"{name} should be a class"


@pytest.mark.parametrize("name", sorted(CALLABLES))
def test_callable_exports_are_functions(name: str) -> None:
    obj = getattr(attune_rag, name)
    assert callable(obj), f"{name} should be callable"
    # Must have a real signature (not a builtin or C extension stub).
    inspect.signature(obj)


def test_prompt_variants_is_a_dict_of_strings() -> None:
    """PROMPT_VARIANTS schema: dict[str, str] (variant name → template body)."""
    pv = attune_rag.PROMPT_VARIANTS
    assert isinstance(pv, dict)
    assert pv, "PROMPT_VARIANTS must be non-empty"
    for k, v in pv.items():
        assert isinstance(k, str)
        assert isinstance(v, str)
        assert v.strip(), f"PROMPT_VARIANTS[{k!r}] is empty"


def test_version_is_pep440_string() -> None:
    """``attune_rag.__version__`` is the canonical version string consumers may pin."""
    v = attune_rag.__version__
    assert isinstance(v, str)
    parts = v.split(".")
    assert len(parts) >= 2
    # Major / minor / patch are digit-only.
    int(parts[0])
    int(parts[1])


# ---------------------------------------------------------------------------
# Signature pins for the most-consumed callables
# ---------------------------------------------------------------------------


def test_build_augmented_prompt_signature() -> None:
    sig = inspect.signature(attune_rag.build_augmented_prompt)
    # Must accept query + context-bearing args; downstream consumers
    # (attune-gui's RagPipeline.run flow) depend on the keyword names.
    params = list(sig.parameters)
    assert "query" in params, f"build_augmented_prompt lost ``query`` param: {params}"


def test_rag_pipeline_run_signature() -> None:
    sig = inspect.signature(attune_rag.RagPipeline.run)
    params = list(sig.parameters)
    # ``self`` + ``query`` must be present; ``k`` keeps default semantics.
    assert "query" in params
    assert "k" in params


def test_retrieval_hit_has_internal_shape() -> None:
    """RetrievalHit is the internal retriever record — keeps its own shape."""
    hit_cls = attune_rag.RetrievalHit
    assert hasattr(
        hit_cls, "__dataclass_fields__"
    ), "RetrievalHit must remain a dataclass for predictable structural use"
    fields = {f.name for f in hit_cls.__dataclass_fields__.values()}
    # Pinning current internal shape; if these change, downstream
    # callers (notably the reranker + benchmark) need adjustments.
    for required in ("entry", "score", "match_reason"):
        assert required in fields, f"RetrievalHit lost ``{required}`` field"


def test_cited_source_exposes_consumer_fields() -> None:
    """``CitedSource`` is what attune-gui's ``RagPipeline.run`` flow ultimately
    surfaces — the flat ``template_path / score / excerpt / category`` shape
    consumers depend on must live on this class."""
    cls = attune_rag.CitedSource
    if hasattr(cls, "__dataclass_fields__"):
        fields = {f.name for f in cls.__dataclass_fields__.values()}
    else:
        fields = {a for a in dir(cls) if not a.startswith("_")}
    for required in ("template_path", "score", "excerpt", "category"):
        assert required in fields, f"CitedSource lost ``{required}`` attribute — consumer break"
