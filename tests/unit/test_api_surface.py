"""Public API surface snapshot test for attune-rag 0.2.0.

Phase 3 of the v1.0 roadmap froze the public surface — see
``docs/specs/api-v0.2-public-surface/`` for the spec. This test pins
two things:

1. ``__all__`` membership for every PUBLIC module. Adding or removing
   a name forces a deliberate update to the ``EXPECTED_*`` frozensets
   below, which is reviewable in PR.
2. Importability of every PUBLIC submodule, by qualified path. This
   is the contract for downstreams that import qualified (attune-help
   imports ``attune_rag.corpus.attune_help.AttuneHelpCorpus``).

Plus a ``__version__`` shape check.

This is broader than ``test_contracts.py``, which locks only the root
``__all__``. When a symbol is intentionally retired, update the
``EXPECTED_*`` constant here AND follow the deprecation policy in
``docs/POLICY.md`` (lands in M4 of Phase 3).
"""

from __future__ import annotations

import importlib
import re

import pytest

import attune_rag

EXPECTED_ROOT_ALL = frozenset(
    {
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
    }
)

EXPECTED_CORPUS_ALL = frozenset(
    {
        "AliasInfo",
        "AttuneHelpCorpus",
        "CorpusProtocol",
        "DirectoryCorpus",
        "DuplicateAliasError",
        "RetrievalEntry",
        "load_aliases_from_file",
    }
)

EXPECTED_PROVIDERS_ALL = frozenset(
    {
        "LLMProvider",
        "get_provider",
        "list_available",
    }
)

EXPECTED_MEASURE_CORPUS_ALL = frozenset({"measure", "MeasureResult"})

EXPECTED_EDITOR_ALL = frozenset(
    {
        "Diagnostic",
        "FileEdit",
        "FileMove",
        "Hunk",
        "Reference",
        "ReferenceContext",
        "ReferenceKind",
        "RenameCollisionError",
        "RenameError",
        "RenamePlan",
        "SchemaError",
        "Severity",
        "apply_rename",
        "autocomplete_aliases",
        "autocomplete_tags",
        "find_references",
        "lint_template",
        "load_schema",
        "parse_frontmatter",
        "plan_rename",
        "validate_frontmatter",
    }
)

PUBLIC_SUBMODULES = (
    "attune_rag.corpus",
    "attune_rag.corpus.attune_help",
    "attune_rag.corpus.help_adapter",
    "attune_rag.providers",
    "attune_rag.measure_corpus",
    "attune_rag.editor",
    "attune_rag.editor.rename",
    "attune_rag.editor.schema",
    "attune_rag.editor.lint",
    "attune_rag.editor.autocomplete",
    "attune_rag.editor.references",
)

MODULE_ALL_EXPECTATIONS = (
    ("attune_rag", EXPECTED_ROOT_ALL),
    ("attune_rag.corpus", EXPECTED_CORPUS_ALL),
    ("attune_rag.providers", EXPECTED_PROVIDERS_ALL),
    ("attune_rag.measure_corpus", EXPECTED_MEASURE_CORPUS_ALL),
    ("attune_rag.editor", EXPECTED_EDITOR_ALL),
)


@pytest.mark.parametrize(("module_path", "expected"), MODULE_ALL_EXPECTATIONS)
def test_module_all_is_frozen(module_path: str, expected: frozenset[str]) -> None:
    """``__all__`` matches the design.md table for every PUBLIC module."""
    module = importlib.import_module(module_path)
    actual = frozenset(module.__all__)
    assert actual == expected, (
        f"{module_path}.__all__ drifted from the frozen surface.\n"
        f"  added:   {sorted(actual - expected)}\n"
        f"  removed: {sorted(expected - actual)}\n"
        f"If intentional, update docs/specs/api-v0.2-public-surface/design.md "
        f"and this test in the same PR. Follow docs/POLICY.md for removals."
    )


@pytest.mark.parametrize("path", PUBLIC_SUBMODULES)
def test_public_submodule_is_importable(path: str) -> None:
    """Downstreams import these by qualified path; the path is the contract."""
    importlib.import_module(path)


def test_root_has_version() -> None:
    """``__version__`` must be present and look like a PEP 440 release."""
    version = getattr(attune_rag, "__version__", None)
    assert version is not None, "attune_rag.__version__ missing"
    assert re.fullmatch(
        r"\d+\.\d+\.\d+(?:[-.]\w+)?", version
    ), f"__version__ {version!r} doesn't match expected PEP-440 shape"


def test_attune_help_corpus_is_importable_qualified() -> None:
    """attune-help imports this exact path — make the contract explicit."""
    from attune_rag.corpus.attune_help import AttuneHelpCorpus

    assert AttuneHelpCorpus.__module__ == "attune_rag.corpus.attune_help"


def test_help_corpus_adapter_protocol_is_importable_qualified() -> None:
    """attune-help references this Protocol by qualified path."""
    from attune_rag.corpus.help_adapter import HelpCorpusAdapter

    assert HelpCorpusAdapter.__module__ == "attune_rag.corpus.help_adapter"
