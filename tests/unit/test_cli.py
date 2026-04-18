"""Unit tests for the attune-rag CLI."""

from __future__ import annotations

import json
from collections.abc import Iterable
from unittest.mock import patch

import pytest

from attune_rag import RetrievalEntry
from attune_rag.cli import main
from attune_rag.corpus.base import CorpusProtocol


class _FakeCorpus(CorpusProtocol):
    def __init__(self) -> None:
        self._entries = {
            "concepts/alpha.md": RetrievalEntry(
                path="concepts/alpha.md",
                category="concepts",
                content="security audit scans for vulnerabilities",
                summary="run a security audit",
            ),
            "concepts/beta.md": RetrievalEntry(
                path="concepts/beta.md",
                category="concepts",
                content="totally unrelated poetry",
            ),
        }

    def entries(self) -> Iterable[RetrievalEntry]:
        return tuple(self._entries.values())

    def get(self, path: str) -> RetrievalEntry | None:
        return self._entries.get(path)

    @property
    def name(self) -> str:
        return "fake-corpus"

    @property
    def version(self) -> str:
        return "1.0.0"


@pytest.fixture(autouse=True)
def _patch_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CLI to use an in-memory corpus so tests don't hit attune-help."""
    from attune_rag import pipeline

    monkeypatch.setattr(pipeline.RagPipeline, "_default_corpus", staticmethod(_FakeCorpus))


def test_query_subcommand_prints_hits(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["query", "security audit"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "concepts/alpha.md" in out
    assert "## Sources" in out


def test_query_no_match_prints_fallback(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["query", "zzzzzzz asdfgh qqqqqq"])
    assert rc == 0
    assert "No grounding context" in capsys.readouterr().out


def _extract_json(text: str) -> dict:
    """Pull the JSON block out of captured output.

    structlog may emit log lines to stdout before the JSON
    payload; the CLI's JSON block always starts at the first
    ``{`` character.
    """
    start = text.find("{")
    assert start >= 0, f"no JSON in output: {text!r}"
    return json.loads(text[start:])


def test_query_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["query", "security audit", "--json"])
    assert rc == 0
    payload = _extract_json(capsys.readouterr().out)
    assert payload["fallback_used"] is False
    assert "hits" in payload
    assert payload["hits"]


def test_query_respects_k(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["query", "security audit", "--json", "-k", "1"])
    assert rc == 0
    payload = _extract_json(capsys.readouterr().out)
    assert len(payload["hits"]) <= 1


def test_corpus_info_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["corpus-info"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "fake-corpus" in out
    assert "Entries: 2" in out


def test_providers_subcommand_runs(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["providers"])
    assert rc == 0
    out = capsys.readouterr().out
    # Output varies by installed extras; just check something printed.
    assert out.strip()


def test_query_with_provider_calls_llm(capsys: pytest.CaptureFixture[str]) -> None:
    class FakeProvider:
        name = "fake"

        async def generate(self, prompt, model=None, max_tokens=2048):  # noqa: ARG002
            return "mocked llm response"

    with patch("attune_rag.providers.get_provider", return_value=FakeProvider()):
        rc = main(["query", "security audit", "--provider", "claude"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mocked llm response" in out
    assert "## Sources" in out


def test_unknown_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main(["not-a-command"])
