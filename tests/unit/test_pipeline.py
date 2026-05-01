"""Unit tests for RagPipeline orchestrator."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

from attune_rag import DirectoryCorpus, RagPipeline, RagResult, RetrievalEntry
from attune_rag.corpus.base import CorpusProtocol


class FakeCorpus(CorpusProtocol):
    def __init__(self, entries: Iterable[RetrievalEntry]) -> None:
        self._entries = {e.path: e for e in entries}

    def entries(self) -> Iterable[RetrievalEntry]:
        return tuple(self._entries.values())

    def get(self, path: str) -> RetrievalEntry | None:
        return self._entries.get(path)

    @property
    def name(self) -> str:
        return "fake"

    @property
    def version(self) -> str:
        return "0"


@pytest.fixture
def corpus() -> FakeCorpus:
    return FakeCorpus(
        [
            RetrievalEntry(
                path="concepts/security-audit.md",
                category="concepts",
                content="Security audit scans for vulnerabilities.",
                summary="Run a security audit",
            ),
            RetrievalEntry(
                path="concepts/smart-test.md",
                category="concepts",
                content="Smart test generates tests for untested code.",
                summary="Generate tests",
            ),
        ]
    )


def test_happy_path_returns_populated_rag_result(corpus: FakeCorpus) -> None:
    pipeline = RagPipeline(corpus=corpus)
    result = pipeline.run("security audit")
    assert isinstance(result, RagResult)
    assert "security audit" in result.augmented_prompt.lower()
    assert "### CONTEXT" in result.augmented_prompt
    assert result.citation.hits
    assert result.fallback_used is False
    assert result.confidence > 0
    assert result.elapsed_ms >= 0


def test_no_match_uses_fallback_prompt(corpus: FakeCorpus) -> None:
    pipeline = RagPipeline(corpus=corpus)
    result = pipeline.run("zzzzzzz asdfgh qwerty unrelated")
    assert result.fallback_used is True
    assert result.confidence == 0.0
    assert "No grounding context" in result.augmented_prompt
    assert result.citation.hits == ()


def test_pipeline_with_directory_corpus(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    (root / "concepts").mkdir(parents=True)
    (root / "concepts" / "security-audit.md").write_text(
        "# security audit\nScans for vulnerabilities."
    )
    (root / "concepts" / "other.md").write_text("# other\nUnrelated content.")
    pipeline = RagPipeline(corpus=DirectoryCorpus(root))
    result = pipeline.run("security audit")
    assert result.fallback_used is False
    assert any("security-audit" in h.template_path for h in result.citation.hits)


def test_structlog_event_emitted(corpus: FakeCorpus, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    import structlog

    structlog.configure(
        processors=[structlog.processors.KeyValueRenderer()],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    with caplog.at_level(logging.INFO, logger="attune_rag.pipeline"):
        RagPipeline(corpus=corpus).run("security audit")
    assert any("rag.run" in r.getMessage() for r in caplog.records)


def test_accepts_injected_retriever(corpus: FakeCorpus) -> None:
    class NoOpRetriever:
        MIN_SCORE = 1.0

        def retrieve(self, query, corpus, k=3):
            return []

    pipeline = RagPipeline(corpus=corpus, retriever=NoOpRetriever())
    result = pipeline.run("security audit")
    assert result.fallback_used is True


def test_default_corpus_raises_when_attune_help_missing() -> None:
    """Verifies the helpful error when no corpus is passed and
    attune-help is absent. sys.modules sentinel works across
    Python 3.10-3.13 (the deprecated MetaPathFinder find_module
    API stopped firing in 3.12+)."""
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key in {"attune_help", "attune_rag.corpus.attune_help"} or key.startswith(
            "attune_help."
        ):
            saved[key] = sys.modules.pop(key)

    sys.modules["attune_help"] = None  # type: ignore[assignment]

    try:
        pipeline = RagPipeline()
        with pytest.raises(RuntimeError, match=r"\[attune-help\]"):
            pipeline.run("any query")
    finally:
        sys.modules.pop("attune_help", None)
        sys.modules.update(saved)


def test_default_corpus_lazy(corpus: FakeCorpus) -> None:
    """Passing a corpus skips default-corpus resolution entirely."""
    pipeline = RagPipeline(corpus=corpus)
    # Does not raise even if attune-help were unavailable.
    result = pipeline.run("security audit")
    assert result.fallback_used is False


def test_confidence_is_bounded(corpus: FakeCorpus) -> None:
    pipeline = RagPipeline(corpus=corpus)
    result = pipeline.run("security audit")
    assert 0.0 <= result.confidence <= 1.0


def test_k_propagated_to_retriever(corpus: FakeCorpus) -> None:
    pipeline = RagPipeline(corpus=corpus)
    result = pipeline.run("security audit smart test", k=1)
    assert len(result.citation.hits) <= 1


def test_default_variant_is_citation(corpus: FakeCorpus) -> None:
    """Pin the default to `citation` — selected by A/B sweep on 2026-04-19
    (46.7% → 6.7% hallucination rate). Under 0.1.5 each passage is
    wrapped in a bare <passage>...</passage> sentinel (for injection
    defense) while the pre-0.1.5 `[P1] source: <path>` header stays
    as the first line inside the tag (for citation-training fidelity)."""
    pipeline = RagPipeline(corpus=corpus)
    result = pipeline.run("security audit")
    assert "<passage>" in result.context
    assert "[P1] source:" in result.context
    assert "<passage>" in result.augmented_prompt
    # The regressed XML-attribute citation shape must stay gone.
    assert 'id="P1"' not in result.augmented_prompt


def test_baseline_variant_opt_in(corpus: FakeCorpus) -> None:
    """The baseline variant uses <passage> sentinels around the
    pre-0.1.5 [source: <path>] header (no P1 numbering)."""
    pipeline = RagPipeline(corpus=corpus)
    result = pipeline.run("security audit", prompt_variant="baseline")
    assert "<passage>" in result.augmented_prompt
    assert "[source:" in result.augmented_prompt
    assert "[P1]" not in result.augmented_prompt


def test_run_and_generate_with_provider_instance(corpus: FakeCorpus) -> None:
    """run_and_generate with an LLMProvider instance calls its generate()."""
    from unittest.mock import AsyncMock

    pipeline = RagPipeline(corpus=corpus)

    class FakeProvider:
        name = "fake"
        generate = AsyncMock(return_value="LLM answer")

    import asyncio

    provider = FakeProvider()
    response, result = asyncio.run(pipeline.run_and_generate("security audit", provider=provider))
    assert response == "LLM answer"
    assert result.fallback_used is False
    provider.generate.assert_awaited_once()
    passed_prompt = (
        provider.generate.await_args.kwargs["prompt"]
        if "prompt" in provider.generate.await_args.kwargs
        else provider.generate.await_args.args[0]
    )
    assert "### CONTEXT" in passed_prompt


def test_run_and_generate_passes_cached_prefix_when_long_enough() -> None:
    """A retrieval result whose augmented prompt exceeds the
    1024-char threshold must arrive at the provider with a
    populated ``cached_prefix``, ending at the USER REQUEST
    marker. This is what gives Anthropic something stable to
    cache across calls.
    """
    from unittest.mock import AsyncMock

    # Stuff the corpus with a single bulky entry so the
    # rendered context comfortably exceeds 1024 chars.
    bulky = "Security audit scans for vulnerabilities. " * 50
    corpus = FakeCorpus(
        [
            RetrievalEntry(
                path="concepts/security-audit.md",
                category="concepts",
                content=bulky,
                summary="Run a security audit",
            ),
        ]
    )
    pipeline = RagPipeline(corpus=corpus)

    class FakeProvider:
        name = "fake"
        generate = AsyncMock(return_value="LLM answer")

    import asyncio

    provider = FakeProvider()
    asyncio.run(pipeline.run_and_generate("security audit", provider=provider))

    kwargs = provider.generate.await_args.kwargs
    assert "cached_prefix" in kwargs
    cached_prefix = kwargs["cached_prefix"]
    assert cached_prefix is not None
    assert cached_prefix.endswith("\n### USER REQUEST\n")
    assert len(cached_prefix) >= 1024


def test_run_and_generate_omits_cached_prefix_when_too_short(
    corpus: FakeCorpus,
) -> None:
    """Below the 1024-char threshold the cached_prefix must be
    None — adding cache_control to a tiny block wastes the
    marker without buying any future hit.
    """
    from unittest.mock import AsyncMock

    pipeline = RagPipeline(corpus=corpus)

    class FakeProvider:
        name = "fake"
        generate = AsyncMock(return_value="ok")

    import asyncio

    provider = FakeProvider()
    asyncio.run(pipeline.run_and_generate("security audit", provider=provider))

    assert provider.generate.await_args.kwargs.get("cached_prefix") is None


def test_run_and_generate_with_provider_name(corpus: FakeCorpus) -> None:
    """A string provider name is dispatched via providers.get_provider."""
    from unittest.mock import AsyncMock, patch

    pipeline = RagPipeline(corpus=corpus)
    fake = type("FP", (), {"name": "fake", "generate": AsyncMock(return_value="ok")})()

    import asyncio

    with patch("attune_rag.providers.get_provider", return_value=fake):
        response, result = asyncio.run(
            pipeline.run_and_generate("security audit", provider="claude")
        )
    assert response == "ok"
    assert result is not None
