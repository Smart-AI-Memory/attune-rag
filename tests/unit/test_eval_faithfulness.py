"""Unit tests for the FaithfulnessJudge.

The Anthropic client is mocked — these tests do not make
real API calls. They verify the judge shapes the request
correctly and parses the tool-use response correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from attune_rag.eval import FaithfulnessJudge, FaithfulnessResult


@dataclass
class _FakeToolUseBlock:
    type: str
    input: dict[str, Any]


@dataclass
class _FakeResponse:
    content: list[Any]


class _FakeMessages:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.last_call: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_call = kwargs
        return _FakeResponse(content=[_FakeToolUseBlock(type="tool_use", input=self._payload)])


class _FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.messages = _FakeMessages(payload)


def _make_judge(payload: dict[str, Any]) -> tuple[FaithfulnessJudge, _FakeClient]:
    client = _FakeClient(payload)
    judge = FaithfulnessJudge(client=client, model="claude-sonnet-4-6")
    return judge, client


@pytest.mark.asyncio
async def test_score_mixed_claims_computes_fraction() -> None:
    judge, _ = _make_judge(
        {
            "supported_claims": ["claim A", "claim B", "claim C"],
            "unsupported_claims": ["claim D"],
            "reasoning": "one hallucination about a nonexistent flag",
        }
    )
    result = await judge.score(
        query="how do I X?",
        answer="Do A, B, C, and the --fake-flag.",
        passages="A and B and C are documented.",
    )
    assert isinstance(result, FaithfulnessResult)
    assert result.score == pytest.approx(3 / 4)
    assert result.total_claims == 4
    assert len(result.supported_claims) == 3
    assert len(result.unsupported_claims) == 1
    assert "hallucination" in result.reasoning


@pytest.mark.asyncio
async def test_score_all_supported_is_perfect() -> None:
    judge, _ = _make_judge(
        {
            "supported_claims": ["claim A", "claim B"],
            "unsupported_claims": [],
            "reasoning": "fully grounded",
        }
    )
    result = await judge.score("q", "a", "p")
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_score_all_unsupported_is_zero() -> None:
    judge, _ = _make_judge(
        {
            "supported_claims": [],
            "unsupported_claims": ["claim A", "claim B"],
            "reasoning": "entirely hallucinated",
        }
    )
    result = await judge.score("q", "a", "p")
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_refusal_answer_scores_perfect() -> None:
    """Zero claims (a refusal) is the correct behavior."""
    judge, _ = _make_judge(
        {
            "supported_claims": [],
            "unsupported_claims": [],
            "reasoning": "answer refused to respond; no claims to evaluate",
        }
    )
    result = await judge.score(
        "unsupported question",
        "The provided context does not cover this question.",
        "unrelated passages",
    )
    assert result.score == 1.0
    assert result.total_claims == 0


@pytest.mark.asyncio
async def test_empty_answer_shortcircuits_without_api_call() -> None:
    judge, client = _make_judge({"supported_claims": [], "unsupported_claims": [], "reasoning": ""})
    result = await judge.score("q", "", "passages")
    assert result.score == 1.0
    assert client.messages.last_call is None


@pytest.mark.asyncio
async def test_passages_list_joined_before_send() -> None:
    judge, client = _make_judge(
        {"supported_claims": ["x"], "unsupported_claims": [], "reasoning": "ok"}
    )
    await judge.score(
        "q",
        "answer claim x",
        passages=["passage one", "passage two", "passage three"],
    )
    sent = client.messages.last_call
    assert sent is not None
    user_msg = sent["messages"][0]["content"]
    assert "passage one" in user_msg
    assert "passage two" in user_msg
    assert "passage three" in user_msg


@pytest.mark.asyncio
async def test_forces_the_report_tool() -> None:
    judge, client = _make_judge(
        {"supported_claims": [], "unsupported_claims": [], "reasoning": "ok"}
    )
    await judge.score("q", "a", "p")
    sent = client.messages.last_call
    assert sent is not None
    assert sent["tool_choice"] == {"type": "tool", "name": "report_faithfulness"}
    tool_names = [t["name"] for t in sent["tools"]]
    assert "report_faithfulness" in tool_names


@pytest.mark.asyncio
async def test_missing_tool_use_block_raises() -> None:
    # Client returns content without a tool_use block.
    class _BadMessages:
        async def create(self, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse(content=[])

    class _BadClient:
        messages = _BadMessages()

    judge = FaithfulnessJudge(client=_BadClient())
    with pytest.raises(RuntimeError, match="did not emit a tool_use block"):
        await judge.score("q", "nonempty answer", "p")


def test_faithfulness_result_to_dict_roundtrips() -> None:
    r = FaithfulnessResult(
        score=0.5,
        supported_claims=["a"],
        unsupported_claims=["b"],
        reasoning="half",
        model="claude-sonnet-4-6",
    )
    d = r.to_dict()
    assert d["score"] == 0.5
    assert d["total_claims"] == 2
    assert d["supported_claims"] == ["a"]
    assert d["unsupported_claims"] == ["b"]
    assert d["reasoning"] == "half"
    assert d["model"] == "claude-sonnet-4-6"


# --- Constructor validation (added in 0.1.4) ---


def test_constructor_rejects_client_and_api_key_together() -> None:
    """Ambiguous: which config should the judge use?"""
    import pytest as _pytest

    from attune_rag.eval.faithfulness import FaithfulnessJudge

    client = _FakeClient({"supported_claims": [], "unsupported_claims": [], "reasoning": ""})
    with _pytest.raises(ValueError, match="not both"):
        FaithfulnessJudge(client=client, api_key="sk-fake")  # pragma: allowlist secret


def test_constructor_without_anthropic_raises_helpful_error() -> None:
    """Simulate the [claude] extra being uninstalled."""
    import sys

    import pytest as _pytest

    from attune_rag.eval.faithfulness import FaithfulnessJudge

    # sys.modules sentinel pattern — see project CLAUDE.md lesson
    # "MetaPathFinder find_module is dead in Python 3.12+".
    saved = sys.modules.get("anthropic")
    sys.modules["anthropic"] = None  # type: ignore[assignment]
    try:
        with _pytest.raises(RuntimeError, match=r"\[claude\] extra"):
            FaithfulnessJudge()
    finally:
        if saved is not None:
            sys.modules["anthropic"] = saved
        else:
            sys.modules.pop("anthropic", None)


# --- Schema validation on tool-use payload (added in 0.1.4) ---


@pytest.mark.asyncio
async def test_score_raises_on_non_list_supported_claims() -> None:
    """A future SDK shape change where supported_claims is not a
    list must surface a clear error, not a cryptic TypeError from
    ``len()``."""
    judge, _ = _make_judge(
        {
            "supported_claims": "not a list",  # shape violation
            "unsupported_claims": [],
            "reasoning": "ok",
        }
    )
    with pytest.raises(RuntimeError, match=r"supported_claims"):
        await judge.score("q", "answer with a claim", "p")


@pytest.mark.asyncio
async def test_score_raises_on_non_string_reasoning() -> None:
    judge, _ = _make_judge(
        {
            "supported_claims": [],
            "unsupported_claims": [],
            "reasoning": {"unexpected": "object"},
        }
    )
    with pytest.raises(RuntimeError, match=r"reasoning"):
        await judge.score("q", "answer", "p")


@pytest.mark.asyncio
async def test_score_coerces_non_string_claim_items_to_strings() -> None:
    """Model occasionally emits non-string items in the claim lists
    (e.g. numbers). The parser should coerce rather than crash, so
    a benchmark doesn't abort on one weird query."""
    judge, _ = _make_judge(
        {
            "supported_claims": [123, "a real claim"],
            "unsupported_claims": [],
            "reasoning": "ok",
        }
    )
    result = await judge.score("q", "a", "p")
    assert len(result.supported_claims) == 2
    assert "123" in result.supported_claims
    assert "a real claim" in result.supported_claims
