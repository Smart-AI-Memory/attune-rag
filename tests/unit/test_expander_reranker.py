"""Unit tests for QueryExpander and LLMReranker."""

from __future__ import annotations

from collections.abc import Iterable
from unittest.mock import MagicMock, patch

import pytest

from attune_rag import LLMReranker, QueryExpander, RagPipeline, RetrievalEntry
from attune_rag.corpus.base import CorpusProtocol
from attune_rag.retrieval import RetrievalHit

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_entry(path: str, summary: str = "", content: str = "content") -> RetrievalEntry:
    category = path.split("/")[0] if "/" in path else ""
    return RetrievalEntry(path=path, category=category, content=content, summary=summary)


def _make_hit(path: str, score: float = 5.0, summary: str = "") -> RetrievalHit:
    return RetrievalHit(entry=_make_entry(path, summary=summary), score=score, match_reason="test")


def _fake_response(text: str) -> MagicMock:
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


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


# ---------------------------------------------------------------------------
# QueryExpander tests
# ---------------------------------------------------------------------------


class TestQueryExpander:
    def _expander_with_mock(self, response_text: str) -> tuple[QueryExpander, MagicMock]:
        expander = QueryExpander(cache=False)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_response(response_text)
        expander._client = mock_client
        return expander, mock_client

    def test_returns_parsed_list(self):
        expander, mock_client = self._expander_with_mock(
            '["release preparation", "publish package", "ship to registry"]'
        )
        result = expander.expand("publish to PyPI")
        assert result == ["release preparation", "publish package", "ship to registry"]
        mock_client.messages.create.assert_called_once()

    def test_returns_empty_on_json_error(self):
        expander, _ = self._expander_with_mock("not valid json")
        assert expander.expand("any query") == []

    def test_returns_empty_on_non_list_json(self):
        expander, _ = self._expander_with_mock('{"key": "value"}')
        assert expander.expand("any query") == []

    def test_returns_empty_on_api_exception(self):
        expander = QueryExpander(cache=False)
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API error")
        expander._client = mock_client
        assert expander.expand("any query") == []

    def test_cache_avoids_second_api_call(self):
        expander = QueryExpander(cache=True)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_response('["alt phrasing"]')
        expander._client = mock_client

        first = expander.expand("cached query")
        second = expander.expand("cached query")
        assert first == second == ["alt phrasing"]
        assert mock_client.messages.create.call_count == 1

    def test_cache_disabled_calls_api_each_time(self):
        expander = QueryExpander(cache=False)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_response('["alt"]')
        expander._client = mock_client

        expander.expand("q")
        expander.expand("q")
        assert mock_client.messages.create.call_count == 2

    def test_uses_prompt_caching_header(self):
        expander = QueryExpander(cache=False)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_response("[]")
        expander._client = mock_client

        expander.expand("test query")
        # system is passed as kwarg
        system = mock_client.messages.create.call_args.kwargs["system"]
        assert any(
            isinstance(block, dict) and block.get("cache_control") == {"type": "ephemeral"}
            for block in system
        )

    def test_requires_claude_extra(self):
        expander = QueryExpander(cache=False)
        with patch.dict("sys.modules", {"anthropic": None}):
            expander._client = None
            with pytest.raises(RuntimeError, match=r"\[claude\]"):
                expander._anthropic  # noqa: B018


# ---------------------------------------------------------------------------
# LLMReranker tests
# ---------------------------------------------------------------------------


class TestLLMReranker:
    def _reranker_with_mock(self, response_text: str) -> tuple[LLMReranker, MagicMock]:
        reranker = LLMReranker()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_response(response_text)
        reranker._client = mock_client
        return reranker, mock_client

    def test_reorders_hits_by_llm_ranking(self):
        hits = [
            _make_hit("a.md", score=10.0),
            _make_hit("b.md", score=8.0),
            _make_hit("c.md", score=6.0),
        ]
        reranker, _ = self._reranker_with_mock("[2, 0, 1]")
        result = reranker.rerank("query", hits)
        assert [h.entry.path for h in result] == ["c.md", "a.md", "b.md"]

    def test_returns_original_on_json_error(self):
        hits = [_make_hit("a.md"), _make_hit("b.md")]
        reranker, _ = self._reranker_with_mock("invalid json")
        result = reranker.rerank("query", hits)
        assert [h.entry.path for h in result] == ["a.md", "b.md"]

    def test_returns_original_on_api_exception(self):
        hits = [_make_hit("a.md"), _make_hit("b.md")]
        reranker = LLMReranker()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        reranker._client = mock_client
        result = reranker.rerank("query", hits)
        assert [h.entry.path for h in result] == ["a.md", "b.md"]

    def test_single_hit_skips_api_call(self):
        hits = [_make_hit("a.md")]
        reranker = LLMReranker()
        mock_client = MagicMock()
        reranker._client = mock_client
        result = reranker.rerank("query", hits)
        assert result == hits
        mock_client.messages.create.assert_not_called()

    def test_appends_missing_indices(self):
        hits = [_make_hit("a.md"), _make_hit("b.md"), _make_hit("c.md")]
        # LLM only mentions indices 2 and 0, skips 1
        reranker, _ = self._reranker_with_mock("[2, 0]")
        result = reranker.rerank("query", hits)
        assert result[0].entry.path == "c.md"
        assert result[1].entry.path == "a.md"
        assert result[2].entry.path == "b.md"

    def test_ignores_out_of_range_indices(self):
        hits = [_make_hit("a.md"), _make_hit("b.md")]
        reranker, _ = self._reranker_with_mock("[99, 1, 0]")
        result = reranker.rerank("query", hits)
        assert [h.entry.path for h in result] == ["b.md", "a.md"]

    def test_uses_prompt_caching_header(self):
        hits = [_make_hit("a.md"), _make_hit("b.md")]
        reranker, mock_client = self._reranker_with_mock("[0, 1]")
        reranker.rerank("query", hits)
        system = mock_client.messages.create.call_args.kwargs["system"]
        assert any(
            isinstance(block, dict) and block.get("cache_control") == {"type": "ephemeral"}
            for block in system
        )


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineWithExpander:
    @pytest.fixture
    def corpus(self) -> FakeCorpus:
        return FakeCorpus(
            [
                _make_entry(
                    "concepts/doc-orchestrator.md",
                    summary="Orchestrates documentation workflows",
                    content="Coordinates documentation generation pipeline.",
                ),
                _make_entry(
                    "concepts/workflow.md",
                    summary="General workflow automation",
                    content="Automates arbitrary workflows.",
                ),
            ]
        )

    def test_expander_joins_phrasings_into_retrieval_query(self, corpus: FakeCorpus):
        expander = QueryExpander(cache=False)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_response('["doc pipeline coordinator"]')
        expander._client = mock_client

        pipeline = RagPipeline(corpus=corpus, expander=expander)
        result = pipeline.run("orchestrate docs")
        # The expansion added "doc pipeline coordinator" so doc-orchestrator
        # should score higher than without expansion.
        assert result.fallback_used is False
        assert result.citation.hits  # at least one hit found

    def test_expander_empty_expansion_falls_back_gracefully(self, corpus: FakeCorpus):
        expander = QueryExpander(cache=False)
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _fake_response("[]")
        expander._client = mock_client

        pipeline = RagPipeline(corpus=corpus, expander=expander)
        result = pipeline.run("orchestrate docs")
        assert not result.fallback_used  # keyword retrieval still runs

    def test_expander_api_failure_falls_back_gracefully(self, corpus: FakeCorpus):
        expander = QueryExpander(cache=False)
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("no network")
        expander._client = mock_client

        pipeline = RagPipeline(corpus=corpus, expander=expander)
        result = pipeline.run("orchestrate docs")
        assert not result.fallback_used


class TestPipelineWithReranker:
    @pytest.fixture
    def corpus(self) -> FakeCorpus:
        return FakeCorpus(
            [
                _make_entry(
                    "concepts/tool-release-prep.md",
                    summary="Preflight checklist before publishing to PyPI.",
                    content="Health security changelog check before release.",
                ),
                _make_entry(
                    "concepts/task-package-publishing.md",
                    summary="Publish a Python package to PyPI step by step.",
                    content="Build upload twine PyPI publish package.",
                ),
                _make_entry(
                    "concepts/tool-other.md",
                    summary="Unrelated tool.",
                    content="Completely unrelated content.",
                ),
            ]
        )

    def test_reranker_retrieves_wider_candidate_set(self, corpus: FakeCorpus):
        reranker = LLMReranker(candidate_multiplier=3)
        mock_client = MagicMock()
        # Return index 0 (tool-release-prep) as top result
        mock_client.messages.create.return_value = _fake_response("[0, 1, 2]")
        reranker._client = mock_client

        pipeline = RagPipeline(corpus=corpus, reranker=reranker)
        result = pipeline.run("publish to PyPI", k=1)
        # With candidate_multiplier=3 and k=1, pipeline retrieves up to 3 candidates
        mock_client.messages.create.assert_called_once()
        assert len(result.citation.hits) == 1

    def test_reranker_reorders_to_promote_preferred_doc(self, corpus: FakeCorpus):
        reranker = LLMReranker()
        mock_client = MagicMock()
        # Keyword retrieval ranks task-package-publishing (idx 0) above
        # tool-release-prep (idx 1). The reranker inverts this: [1, 0].
        mock_client.messages.create.return_value = _fake_response("[1, 0]")
        reranker._client = mock_client

        pipeline = RagPipeline(corpus=corpus, reranker=reranker)
        result = pipeline.run("publish to PyPI", k=2)
        paths = [h.template_path for h in result.citation.hits]
        assert paths[0] == "concepts/tool-release-prep.md"

    def test_reranker_failure_returns_keyword_order(self, corpus: FakeCorpus):
        reranker = LLMReranker()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        reranker._client = mock_client

        pipeline = RagPipeline(corpus=corpus, reranker=reranker)
        result = pipeline.run("publish PyPI package", k=2)
        # Should still return results (in keyword order)
        assert not result.fallback_used
        assert result.citation.hits

    def test_expander_and_reranker_compose(self, corpus: FakeCorpus):
        expander = QueryExpander(cache=False)
        exp_client = MagicMock()
        exp_client.messages.create.return_value = _fake_response('["release package workflow"]')
        expander._client = exp_client

        reranker = LLMReranker()
        rer_client = MagicMock()
        rer_client.messages.create.return_value = _fake_response("[0, 1]")
        reranker._client = rer_client

        pipeline = RagPipeline(corpus=corpus, expander=expander, reranker=reranker)
        result = pipeline.run("publish to PyPI", k=2)
        # Both were called
        exp_client.messages.create.assert_called_once()
        rer_client.messages.create.assert_called_once()
        assert not result.fallback_used
