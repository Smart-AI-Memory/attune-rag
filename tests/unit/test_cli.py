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

        async def generate(
            self, prompt, model=None, max_tokens=2048, cached_prefix=None
        ):  # noqa: ARG002
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


# ── stderr ANSI hardening ──


def test_safe_stderr_strips_ansi_control_chars() -> None:
    from attune_rag.cli import _safe_stderr

    # ESC (\x1b) starts every ANSI escape sequence. A path containing one
    # would, if printed verbatim to a real terminal, repaint or hide the
    # surrounding output.
    msg = "Parent directory does not exist: /tmp/\x1b[31mEVIL\x1b[0m/dash.html"
    cleaned = _safe_stderr(msg)
    assert "\x1b" not in cleaned
    # The byte is replaced with "?" so the message stays readable.
    assert "?[31mEVIL?[0m" in cleaned


def test_safe_stderr_strips_bel_and_backspace() -> None:
    from attune_rag.cli import _safe_stderr

    msg = "boom\x07\x08\x7f trailer"
    cleaned = _safe_stderr(msg)
    for bad in ("\x07", "\x08", "\x7f"):
        assert bad not in cleaned


def test_safe_stderr_preserves_tab_newline_cr() -> None:
    from attune_rag.cli import _safe_stderr

    msg = "line1\nline2\twith tab\r"
    assert _safe_stderr(msg) == msg


def test_dashboard_render_error_stderr_is_sanitized(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    # Trigger _validate_output_path's "Parent directory does not exist"
    # ValueError by aiming at a nonexistent subdirectory whose name
    # carries ANSI escape bytes. The CLI must scrub them before printing.
    bad = str(tmp_path / "missing-\x1b[31mPWN\x1b[0m" / "dash.html")
    rc = main(["dashboard", "render", "--out", bad])
    assert rc == 2
    err = capsys.readouterr().err
    assert "\x1b" not in err
    assert "PWN" in err  # readable form survives


# ---------------------------------------------------------------------------
# Feature-access flags: --corpus-path / --retriever / --min-score /
# --prompt-variant (usability audit 2026-06-10, step 2)
# ---------------------------------------------------------------------------


def _write_md_corpus(root) -> None:
    (root / "concepts").mkdir()
    (root / "concepts" / "security-audit.md").write_text(
        "---\nname: security-audit\n---\n\nRun a security audit to scan for vulnerabilities.",
        encoding="utf-8",
    )


def test_query_corpus_path_uses_directory_corpus(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_md_corpus(tmp_path)
    rc = main(["query", "security audit", "--corpus-path", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "concepts/security-audit.md" in out


def test_query_corpus_path_missing_is_clean_error(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["query", "x", "--corpus-path", str(tmp_path / "nope")])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "not a directory" in err


def test_query_min_score_abstains(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["query", "security audit", "--min-score", "1000"])
    assert rc == 0
    assert "No grounding context found" in capsys.readouterr().out


def test_query_min_score_rejected_for_non_keyword(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["query", "x", "--retriever", "hybrid", "--min-score", "5"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--min-score only applies" in err


def test_query_retriever_hybrid_falls_back_keyword_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Force the embedding leg off so the test never loads a model; the
    # hybrid then degrades to keyword-only order, which is the documented
    # base-install behavior.
    from attune_rag.hybrid import HybridRetriever

    monkeypatch.setattr(HybridRetriever, "_get_embedding", lambda self: None)
    rc = main(["query", "security audit", "--retriever", "hybrid"])
    assert rc == 0
    assert "concepts/alpha.md" in capsys.readouterr().out


def test_query_retriever_transformer_missing_extra_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Simulate the [transformers] extra being absent: the encoder load
    # raises RuntimeError with the install hint, and the CLI must turn
    # that into a one-line error, not a traceback.
    from attune_rag.embedding import EmbeddingRetriever

    def _raise(self):
        raise RuntimeError(
            "TransformerRetriever requires the [transformers] extra. "
            "Install with: pip install 'attune-rag[transformers]'"
        )

    # Patch _corpus_matrix (not just _get_encoder) because it runs first
    # and would `import numpy` — absent on a true base install.
    monkeypatch.setattr(EmbeddingRetriever, "_corpus_matrix", lambda self, corpus: _raise(self))
    rc = main(["query", "security audit", "--retriever", "transformer"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "[transformers] extra" in err
    assert "Traceback" not in err


def test_query_retriever_transformer_with_fake_encoder(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    np = pytest.importorskip("numpy")
    # Patch the SUBCLASS override — TransformerRetriever defines its own
    # _get_encoder, so patching EmbeddingRetriever's would silently fall
    # through to the real sentence-transformers load.
    from attune_rag.transformer import TransformerRetriever

    class FakeEncoder:
        def encode(self, texts):
            return np.ones((len(texts), 2))

    monkeypatch.setattr(TransformerRetriever, "_get_encoder", lambda self: FakeEncoder())
    rc = main(["query", "security audit", "--retriever", "transformer"])
    assert rc == 0
    # All-equal similarities tie-break by path: alpha.md sorts first.
    assert "concepts/alpha.md" in capsys.readouterr().out


def test_query_prompt_variant_passthrough(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from attune_rag import pipeline as pipeline_mod

    captured: dict[str, str] = {}
    orig = pipeline_mod.RagPipeline.run

    def spy(self, query, k=3, prompt_variant="citation"):
        captured["variant"] = prompt_variant
        return orig(self, query, k=k, prompt_variant=prompt_variant)

    monkeypatch.setattr(pipeline_mod.RagPipeline, "run", spy)
    rc = main(["query", "security audit", "--prompt-variant", "strict"])
    assert rc == 0
    assert captured["variant"] == "strict"


def test_corpus_info_corpus_path(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_md_corpus(tmp_path)
    rc = main(["corpus-info", "--corpus-path", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "directory:" in out
    assert "Entries: 1" in out


def test_corpus_info_missing_default_corpus_clean_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With no corpus flag and the [attune-help] extra absent, the CLI
    # must print the actionable install hint cleanly, not a traceback.
    from attune_rag import pipeline as pipeline_mod

    def _unavailable():
        raise RuntimeError(
            "No corpus provided and AttuneHelpCorpus is unavailable. "
            "Either pass a corpus= (e.g. DirectoryCorpus) or install "
            "'attune-rag[attune-help]'."
        )

    monkeypatch.setattr(pipeline_mod.RagPipeline, "_default_corpus", staticmethod(_unavailable))
    rc = main(["corpus-info"])
    assert rc == 2
    assert "attune-rag[attune-help]" in capsys.readouterr().err
