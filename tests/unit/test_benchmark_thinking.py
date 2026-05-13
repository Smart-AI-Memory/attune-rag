"""Unit tests for the ``--thinking`` plumbing in attune-rag-benchmark.

Covers argparse → kwarg translation, env-var defaults, and the
flag-wins-over-env precedence rule. Does not exercise the actual
benchmark loop (which spends API tokens).
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from attune_rag.benchmark import _env_bool, _env_int

# --- env-var helpers ---


def test_env_bool_truthy_values() -> None:
    for value in ("1", "true", "TRUE", "yes", "on"):
        with patch.dict(os.environ, {"X": value}):
            assert _env_bool("X") is True, f"{value!r} should be truthy"


def test_env_bool_falsy_values() -> None:
    for value in ("", "0", "false", "no", "off", "random"):
        with patch.dict(os.environ, {"X": value}):
            assert _env_bool("X") is False, f"{value!r} should be falsy"


def test_env_bool_unset() -> None:
    env = {k: v for k, v in os.environ.items() if k != "X"}
    with patch.dict(os.environ, env, clear=True):
        assert _env_bool("X") is False


def test_env_int_parses_valid() -> None:
    with patch.dict(os.environ, {"X": "12345"}):
        assert _env_int("X") == 12345


def test_env_int_returns_none_on_unset() -> None:
    env = {k: v for k, v in os.environ.items() if k != "X"}
    with patch.dict(os.environ, env, clear=True):
        assert _env_int("X") is None


def test_env_int_returns_none_on_malformed() -> None:
    with patch.dict(os.environ, {"X": "not-a-number"}):
        assert _env_int("X") is None


def test_env_int_returns_none_on_empty_string() -> None:
    with patch.dict(os.environ, {"X": "   "}):
        assert _env_int("X") is None


# --- argparse wiring ---


def _parse_args(argv: list[str]) -> object:
    """Re-build the benchmark argument parser and parse argv.

    Mirrors `main()` but skips the run — we only need the namespace.
    """
    # Import lazily; the function lives in benchmark.main but isn't
    # exported. Build a minimal parser that mirrors the flags we
    # actually test (avoids coupling test to internal structure).
    import argparse

    from attune_rag.benchmark import _env_bool, _env_int

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=_env_bool("ATTUNE_RAG_FAITHFULNESS_THINKING"),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=_env_int("ATTUNE_RAG_FAITHFULNESS_THINKING_BUDGET"),
    )
    return parser.parse_args(argv)


def test_thinking_flag_off_by_default() -> None:
    env = {k: v for k, v in os.environ.items() if k not in {"ATTUNE_RAG_FAITHFULNESS_THINKING"}}
    with patch.dict(os.environ, env, clear=True):
        args = _parse_args([])
        assert args.thinking is False


def test_thinking_flag_explicit() -> None:
    env = {k: v for k, v in os.environ.items() if k not in {"ATTUNE_RAG_FAITHFULNESS_THINKING"}}
    with patch.dict(os.environ, env, clear=True):
        args = _parse_args(["--thinking"])
        assert args.thinking is True


def test_thinking_env_var_sets_default() -> None:
    with patch.dict(os.environ, {"ATTUNE_RAG_FAITHFULNESS_THINKING": "1"}):
        args = _parse_args([])
        assert args.thinking is True


def test_thinking_flag_present_with_env_off() -> None:
    """Flag should still produce True even if env says off — we don't
    support an explicit "off" CLI flag, so flag presence is dominant
    via store_true semantics."""
    with patch.dict(os.environ, {"ATTUNE_RAG_FAITHFULNESS_THINKING": "0"}):
        args = _parse_args(["--thinking"])
        assert args.thinking is True


def test_thinking_budget_env_var_flows_through() -> None:
    with patch.dict(os.environ, {"ATTUNE_RAG_FAITHFULNESS_THINKING_BUDGET": "65536"}):
        args = _parse_args([])
        assert args.thinking_budget == 65536


def test_thinking_budget_cli_overrides_env() -> None:
    with patch.dict(os.environ, {"ATTUNE_RAG_FAITHFULNESS_THINKING_BUDGET": "65536"}):
        args = _parse_args(["--thinking-budget", "16384"])
        assert args.thinking_budget == 16384


def test_thinking_budget_none_when_unset() -> None:
    env = {k: v for k, v in os.environ.items() if k != "ATTUNE_RAG_FAITHFULNESS_THINKING_BUDGET"}
    with patch.dict(os.environ, env, clear=True):
        args = _parse_args([])
        assert args.thinking_budget is None


# --- _score_faithfulness wiring ---


@pytest.mark.asyncio
async def test_score_faithfulness_passes_thinking_kwargs_to_judge(monkeypatch: Any) -> None:
    """The benchmark loop forwards use_thinking + thinking_budget to
    judge.score. Stub the pipeline + judge to capture kwargs."""
    from attune_rag import benchmark as bench_mod

    captured: dict[str, Any] = {}

    class _StubPipeline:
        async def run_and_generate(self, *args: Any, **kwargs: Any) -> Any:
            class _RagResult:
                context = "some passages"
                used_native_citations = False
                claim_citations: list[Any] = []

            return "an answer", _RagResult()

    class _StubJudge:
        def __init__(self) -> None:
            pass

        async def score(self, **kwargs: Any) -> Any:
            captured.update(kwargs)

            class _Verdict:
                score = 1.0
                supported_claims = ["x"]
                unsupported_claims: list[str] = []
                total_claims = 1
                thinking_used = kwargs.get("use_thinking", False)

            return _Verdict()

    import attune_rag

    monkeypatch.setattr(attune_rag, "RagPipeline", _StubPipeline)
    from attune_rag import eval as eval_mod

    monkeypatch.setattr(eval_mod, "FaithfulnessJudge", _StubJudge)

    queries = [{"id": "q1", "query": "?", "difficulty": "easy"}]
    result = await bench_mod._score_faithfulness(
        queries,
        k=3,
        use_thinking=True,
        thinking_budget_tokens=16384,
    )
    assert captured["use_thinking"] is True
    assert captured["thinking_budget_tokens"] == 16384
    assert result["per_query"][0]["thinking_used"] is True


# Need typing.Any in the monkeypatch test
from typing import Any  # noqa: E402  (import at end for the test above)
