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
    with pytest.raises(RuntimeError, match="no tool_use or text block"):
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


# --- Parser fallback paths (added for extended thinking support) ---


@dataclass
class _FakeTextBlock:
    type: str
    text: str


@dataclass
class _FakeThinkingBlock:
    type: str
    thinking: str


class _ScriptedMessages:
    """Like _FakeMessages but returns a pre-built content list."""

    def __init__(self, content_blocks: list[Any]) -> None:
        self._content = content_blocks
        self.last_call: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.last_call = kwargs
        return _FakeResponse(content=list(self._content))


class _ScriptedClient:
    def __init__(self, content_blocks: list[Any]) -> None:
        self.messages = _ScriptedMessages(content_blocks)


@pytest.mark.asyncio
async def test_parser_falls_back_to_text_block_when_no_tool_use() -> None:
    """When the model declines to call the tool, parser reads JSON
    from a text block instead."""
    payload = {
        "supported_claims": ["a", "b"],
        "unsupported_claims": ["c"],
        "reasoning": "two of three supported",
    }
    import json as _json

    client = _ScriptedClient([_FakeTextBlock(type="text", text=_json.dumps(payload))])
    judge = FaithfulnessJudge(client=client)
    result = await judge.score("q", "answer", "p")
    assert result.score == pytest.approx(2 / 3)
    assert "two of three" in result.reasoning


@pytest.mark.asyncio
async def test_parser_strips_json_code_fences() -> None:
    """Thinking-mode responses sometimes wrap JSON in ```json fences."""
    fenced = (
        "```json\n"
        + '{"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"}'
        + "\n```"
    )
    client = _ScriptedClient([_FakeTextBlock(type="text", text=fenced)])
    judge = FaithfulnessJudge(client=client)
    result = await judge.score("q", "a", "p")
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_parser_raises_on_unparseable_text_block() -> None:
    client = _ScriptedClient([_FakeTextBlock(type="text", text="not json at all, just prose")])
    judge = FaithfulnessJudge(client=client)
    with pytest.raises(RuntimeError, match="unparseable text"):
        await judge.score("q", "a", "p")


@pytest.mark.asyncio
async def test_parser_raises_when_text_json_not_object() -> None:
    """Text block parses to a JSON array, not an object."""
    client = _ScriptedClient([_FakeTextBlock(type="text", text='["a", "b"]')])
    judge = FaithfulnessJudge(client=client)
    with pytest.raises(RuntimeError, match="not an object"):
        await judge.score("q", "a", "p")


@pytest.mark.asyncio
async def test_parser_skips_thinking_blocks() -> None:
    """thinking blocks must be ignored even when present."""
    payload = {
        "supported_claims": ["claim"],
        "unsupported_claims": [],
        "reasoning": "ok",
    }
    client = _ScriptedClient(
        [
            _FakeThinkingBlock(type="thinking", thinking="long reasoning..."),
            _FakeToolUseBlock(type="tool_use", input=payload),
        ]
    )
    judge = FaithfulnessJudge(client=client)
    result = await judge.score("q", "a", "p")
    assert result.score == 1.0
    assert result.supported_claims == ["claim"]


@pytest.mark.asyncio
async def test_parser_prefers_tool_use_over_text_block() -> None:
    """When both block types appear, tool_use wins (schema-guaranteed)."""
    tool_payload = {
        "supported_claims": ["from_tool"],
        "unsupported_claims": [],
        "reasoning": "tool",
    }
    text_payload = (
        '{"supported_claims": ["from_text"], "unsupported_claims": [], "reasoning": "text"}'
    )
    client = _ScriptedClient(
        [
            _FakeTextBlock(type="text", text=text_payload),
            _FakeToolUseBlock(type="tool_use", input=tool_payload),
        ]
    )
    judge = FaithfulnessJudge(client=client)
    result = await judge.score("q", "a", "p")
    assert result.supported_claims == ["from_tool"]


# --- Extended thinking on score() (added in 0.1.15) ---


@pytest.mark.asyncio
async def test_score_without_thinking_omits_thinking_block() -> None:
    """Back-compat: default call shape must not include `thinking` and
    must keep tool_choice forced to the report tool."""
    judge, client = _make_judge(
        {"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"}
    )
    await judge.score("q", "a", "p")
    sent = client.messages.last_call
    assert sent is not None
    assert "thinking" not in sent
    assert sent["tool_choice"] == {"type": "tool", "name": "report_faithfulness"}


@pytest.mark.asyncio
async def test_score_with_thinking_sends_thinking_block_and_auto_tool_choice() -> None:
    judge, client = _make_judge(
        {"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"}
    )
    await judge.score("q", "a", "p", use_thinking=True)
    sent = client.messages.last_call
    assert sent is not None
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 32768}
    # Anthropic constraint: thinking + tools requires tool_choice in
    # {"auto", "none"}.
    assert sent["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_score_with_thinking_bumps_max_tokens_above_budget() -> None:
    """Anthropic constraint: max_tokens must exceed thinking
    budget_tokens, since it caps the combined output in thinking
    mode. The library adds thinking_budget on top of the caller's
    max_tokens so the caller's value keeps its original meaning."""
    judge, client = _make_judge(
        {"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"}
    )
    await judge.score(
        "q",
        "a",
        "p",
        max_tokens=2048,
        use_thinking=True,
        thinking_budget_tokens=32768,
    )
    sent = client.messages.last_call
    assert sent is not None
    # 2048 (caller's reply budget) + 32768 (thinking budget) = 34816.
    assert sent["max_tokens"] == 34816
    # And max_tokens must strictly exceed thinking budget per the
    # API constraint.
    assert sent["max_tokens"] > sent["thinking"]["budget_tokens"]


@pytest.mark.asyncio
async def test_score_without_thinking_passes_max_tokens_unchanged() -> None:
    """Back-compat: in non-thinking mode max_tokens is sent verbatim."""
    judge, client = _make_judge(
        {"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"}
    )
    await judge.score("q", "a", "p", max_tokens=2048)
    sent = client.messages.last_call
    assert sent is not None
    assert sent["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_score_with_thinking_custom_budget_flows_through() -> None:
    judge, client = _make_judge(
        {"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"}
    )
    await judge.score("q", "a", "p", use_thinking=True, thinking_budget_tokens=65536)
    sent = client.messages.last_call
    assert sent is not None
    assert sent["thinking"]["budget_tokens"] == 65536


@pytest.mark.asyncio
async def test_score_with_thinking_sets_result_field() -> None:
    judge, _ = _make_judge({"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"})
    result = await judge.score("q", "a", "p", use_thinking=True)
    assert result.thinking_used is True


@pytest.mark.asyncio
async def test_score_without_thinking_thinking_used_is_false() -> None:
    judge, _ = _make_judge({"supported_claims": ["a"], "unsupported_claims": [], "reasoning": "ok"})
    result = await judge.score("q", "a", "p")
    assert result.thinking_used is False


def test_to_dict_includes_thinking_used() -> None:
    r = FaithfulnessResult(
        score=1.0,
        supported_claims=["a"],
        unsupported_claims=[],
        reasoning="ok",
        model="claude-sonnet-4-6",
        thinking_used=True,
    )
    d = r.to_dict()
    assert d["thinking_used"] is True


def test_to_dict_thinking_used_defaults_false() -> None:
    """Existing callers constructing FaithfulnessResult without the
    new field get thinking_used=False in to_dict()."""
    r = FaithfulnessResult(
        score=1.0,
        supported_claims=[],
        unsupported_claims=[],
        reasoning="ok",
        model="claude-sonnet-4-6",
    )
    d = r.to_dict()
    assert d["thinking_used"] is False
