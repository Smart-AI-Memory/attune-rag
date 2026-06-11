"""Routing tests for attune_rag.auth + FaithfulnessJudge auth modes.

No real LLM calls: the subscription path is exercised by
monkeypatching ``attune_rag.auth.query_subscription_structured``
(or installing a fake ``claude_agent_sdk`` in ``sys.modules`` for
the adapter-level test), and the API path by injecting a fake
client. Mirrors attune-author's ``tests/test_auth_routing.py``.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from attune_rag import auth
from attune_rag.eval import FaithfulnessJudge

_PAYLOAD = {
    "supported_claims": ["claim A"],
    "unsupported_claims": ["claim B"],
    "reasoning": "one of each",
}


@pytest.fixture(autouse=True)
def _reset_telemetry() -> None:
    auth.reset_auth_telemetry()


def _enable_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make subscription_available() return True without the SDK."""
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.delenv("ATTUNE_RAG_AUTH_MODE", raising=False)
    monkeypatch.setattr(auth, "_sdk_importable", lambda: True)


# ---------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------


def test_auto_without_signals_resolves_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATTUNE_RAG_AUTH_MODE", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    assert auth.resolve_auth_mode() == "api"


def test_auto_with_subscription_resolves_sub(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    assert auth.resolve_auth_mode() == "sub"


def test_claudecode_alone_is_not_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.delenv("ATTUNE_RAG_AUTH_MODE", raising=False)
    monkeypatch.setattr(auth, "_sdk_importable", lambda: False)
    assert auth.subscription_available() is False
    assert auth.resolve_auth_mode() == "api"


def test_forced_api_wins_over_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    assert auth.resolve_auth_mode("api") == "api"


def test_forced_sub_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDECODE", raising=False)
    with pytest.raises(ValueError, match="no subscription session"):
        auth.resolve_auth_mode("sub")


def test_env_var_mode_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    monkeypatch.setenv("ATTUNE_RAG_AUTH_MODE", "api")
    assert auth.resolve_auth_mode() == "api"


def test_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="Invalid auth mode"):
        auth.resolve_auth_mode("subscription")


def test_redact_strips_key_shapes() -> None:
    raw = "boom sk-ant-" + "api03-FAKEFAKEFAKE token"
    assert "sk-ant-<redacted>" in auth._redact(raw)
    assert "FAKEFAKEFAKE" not in auth._redact(raw)


# ---------------------------------------------------------------
# Judge construction
# ---------------------------------------------------------------


def test_judge_sub_route_builds_no_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    judge = FaithfulnessJudge()
    assert judge._route == "sub"
    assert judge._client is None


def test_injected_client_always_api_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    judge = FaithfulnessJudge(client=object())
    assert judge._route == "api"


def test_judge_forced_sub_unavailable_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDECODE", raising=False)
    with pytest.raises(ValueError, match="no subscription session"):
        FaithfulnessJudge(auth_mode="sub")


# ---------------------------------------------------------------
# score() routing
# ---------------------------------------------------------------


async def test_score_routes_subscription(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    seen: dict[str, Any] = {}

    async def fake_query(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return dict(_PAYLOAD)

    monkeypatch.setattr(auth, "query_subscription_structured", fake_query)
    judge = FaithfulnessJudge(model="claude-sonnet-4-6")
    result = await judge.score(query="q?", answer="A and B.", passages="A only.")

    assert result.score == pytest.approx(0.5)
    assert seen["model"] == "claude-sonnet-4-6"
    assert seen["schema"]["required"] == [
        "supported_claims",
        "unsupported_claims",
        "reasoning",
    ]
    assert auth.auth_telemetry() == {"sub_calls": 1.0, "api_calls": 0.0}


async def test_score_sub_ignores_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)

    async def fake_query(**kwargs: Any) -> dict[str, Any]:
        return dict(_PAYLOAD)

    monkeypatch.setattr(auth, "query_subscription_structured", fake_query)
    judge = FaithfulnessJudge()
    result = await judge.score(query="q?", answer="A.", passages="A.", use_thinking=True)
    assert result.thinking_used is False


async def test_score_auto_falls_back_to_api(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    async def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("subscription expired")

    monkeypatch.setattr(auth, "query_subscription_structured", boom)

    @dataclass
    class _FakeBlock:
        type: str
        input: dict[str, Any]

    @dataclass
    class _FakeResponse:
        content: list[Any] = field(default_factory=list)

    class _FakeMessages:
        async def create(self, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse(content=[_FakeBlock(type="tool_use", input=dict(_PAYLOAD))])

    fake_client = types.SimpleNamespace(messages=_FakeMessages())
    judge = FaithfulnessJudge()
    monkeypatch.setattr(judge, "_build_api_client", lambda: fake_client)

    result = await judge.score(query="q?", answer="A and B.", passages="A.")
    assert result.score == pytest.approx(0.5)
    assert auth.auth_telemetry() == {"sub_calls": 0.0, "api_calls": 1.0}


async def test_score_forced_sub_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    async def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("denied for sk-ant-" + "api03-FAKEFAKE")

    monkeypatch.setattr(auth, "query_subscription_structured", boom)
    judge = FaithfulnessJudge(auth_mode="sub")

    with pytest.raises(RuntimeError) as excinfo:
        await judge.score(query="q?", answer="A.", passages="A.")
    assert "sk-ant-<redacted>" in str(excinfo.value)
    assert "FAKEFAKE" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


async def test_score_sub_failure_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_subscription(monkeypatch)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    async def boom(**kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("nope")

    monkeypatch.setattr(auth, "query_subscription_structured", boom)
    judge = FaithfulnessJudge()
    with pytest.raises(RuntimeError, match="nope"):
        await judge.score(query="q?", answer="A.", passages="A.")


async def test_empty_answer_short_circuits_before_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_subscription(monkeypatch)

    async def boom(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("should not be called")

    monkeypatch.setattr(auth, "query_subscription_structured", boom)
    judge = FaithfulnessJudge()
    result = await judge.score(query="q?", answer="   ", passages="A.")
    assert result.score == 1.0
    assert auth.auth_telemetry() == {"sub_calls": 0.0, "api_calls": 0.0}


# ---------------------------------------------------------------
# Adapter-level: query_subscription_structured against a fake SDK
# ---------------------------------------------------------------


async def test_adapter_reads_structured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeOptions:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    class _FakeResultMessage:
        structured_output = dict(_PAYLOAD)

    def fake_query(*, prompt: str, options: Any):
        async def _gen():
            yield _FakeResultMessage()

        captured["prompt"] = prompt
        return _gen()

    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = _FakeOptions  # type: ignore[attr-defined]
    fake_sdk.ResultMessage = _FakeResultMessage  # type: ignore[attr-defined]
    fake_sdk.query = fake_query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    payload = await auth.query_subscription_structured(
        system="sys", user_message="hello", model="m", schema={"type": "object"}
    )
    assert payload == _PAYLOAD
    assert captured["prompt"] == "hello"
    assert captured["setting_sources"] == []
    # 2, not 1 — structured output costs the CLI an extra turn
    assert captured["max_turns"] == 2
    assert captured["output_format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert captured["env"] == {"ATTUNE_RAG_SDK_SUBPROCESS": "1"}


async def test_adapter_raises_without_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeOptions:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class _FakeResultMessage:
        structured_output = None

    def fake_query(*, prompt: str, options: Any):
        async def _gen():
            yield _FakeResultMessage()

        return _gen()

    fake_sdk = types.ModuleType("claude_agent_sdk")
    fake_sdk.ClaudeAgentOptions = _FakeOptions  # type: ignore[attr-defined]
    fake_sdk.ResultMessage = _FakeResultMessage  # type: ignore[attr-defined]
    fake_sdk.query = fake_query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_sdk)

    with pytest.raises(auth.SubscriptionCallError, match="no structured output"):
        await auth.query_subscription_structured(system="s", user_message="u", model="m", schema={})
