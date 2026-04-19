"""Unit tests for prompt assembly helpers."""

from __future__ import annotations

import pytest

from attune_rag import build_augmented_prompt
from attune_rag.corpus.base import RetrievalEntry
from attune_rag.prompts import (
    PROMPT_VARIANTS,
    join_context,
    join_context_numbered,
)
from attune_rag.retrieval import RetrievalHit


def _hit(path: str, content: str, score: float = 1.0) -> RetrievalHit:
    entry = RetrievalEntry(path=path, category="concepts", content=content)
    return RetrievalHit(entry=entry, score=score, match_reason="test")


def test_augmented_prompt_contains_both_sections() -> None:
    out = build_augmented_prompt("how do I run a security audit?", "context body")
    assert "### CONTEXT" in out
    assert "### USER REQUEST" in out
    assert "how do I run a security audit?" in out
    assert "context body" in out


def test_augmented_prompt_has_injection_guard() -> None:
    out = build_augmented_prompt("q", "ctx")
    assert "not as directives" in out or "as data" in out


def test_augmented_prompt_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_augmented_prompt("", "ctx")
    with pytest.raises(ValueError, match="non-empty"):
        build_augmented_prompt("   ", "ctx")


def test_augmented_prompt_strips_whitespace() -> None:
    out = build_augmented_prompt("  query  ", "  context  ")
    assert "  query  " not in out
    assert "query" in out
    assert "context" in out


def test_join_context_concatenates_hits() -> None:
    hits = [
        _hit("a.md", "alpha content"),
        _hit("b.md", "beta content"),
    ]
    ctx = join_context(hits)
    assert "alpha content" in ctx
    assert "beta content" in ctx
    assert "[source: a.md]" in ctx
    assert "[source: b.md]" in ctx


def test_join_context_preserves_source_boundaries() -> None:
    hits = [_hit("a.md", "one"), _hit("b.md", "two")]
    ctx = join_context(hits)
    assert "---" in ctx


def test_join_context_respects_max_chars() -> None:
    big = "x" * 10_000
    hits = [_hit("a.md", big), _hit("b.md", big), _hit("c.md", big)]
    ctx = join_context(hits, max_chars=5_000)
    assert len(ctx) <= 5_000


def test_join_context_empty_hits_returns_empty() -> None:
    assert join_context([]) == ""


# --- Prompt variant tests (experiment 2) ---


def test_prompt_variants_registry_has_four_entries() -> None:
    assert set(PROMPT_VARIANTS) == {"baseline", "strict", "citation", "anti_prior"}


@pytest.mark.parametrize("variant", ["baseline", "strict", "citation", "anti_prior"])
def test_every_variant_renders_both_sections(variant: str) -> None:
    out = build_augmented_prompt("my question", "ctx body", variant=variant)
    assert "my question" in out
    assert "ctx body" in out


def test_strict_variant_forbids_prior_knowledge() -> None:
    out = build_augmented_prompt("q", "ctx", variant="strict")
    assert "ONLY" in out
    # Wrapping may split words; normalize for substring check.
    assert "provided context does not" in " ".join(out.split())


def test_citation_variant_requires_markers() -> None:
    out = build_augmented_prompt("q", "ctx", variant="citation")
    assert "[P1]" in out
    assert "citation" in out.lower() or "cite" in out.lower()


def test_anti_prior_variant_warns_about_training_data() -> None:
    out = build_augmented_prompt("q", "ctx", variant="anti_prior")
    assert "training" in out.lower()


def test_unknown_variant_raises() -> None:
    with pytest.raises(ValueError, match="unknown prompt variant"):
        build_augmented_prompt("q", "ctx", variant="does-not-exist")


def test_join_context_numbered_labels_passages() -> None:
    hits = [_hit("a.md", "alpha"), _hit("b.md", "beta"), _hit("c.md", "gamma")]
    ctx = join_context_numbered(hits)
    assert "[P1] source: a.md" in ctx
    assert "[P2] source: b.md" in ctx
    assert "[P3] source: c.md" in ctx
    assert "alpha" in ctx and "beta" in ctx and "gamma" in ctx


def test_join_context_numbered_empty_returns_empty() -> None:
    assert join_context_numbered([]) == ""
