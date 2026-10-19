"""A missing judge verdict must never become a perfect faithfulness score."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from attune_rag import auth
from attune_rag.benchmark import _transient_primary_provider_error
from attune_rag.eval import FaithfulnessJudge

_VERDICT = {
    "supported_claims": ["Claim A"],
    "unsupported_claims": ["Claim B"],
    "reasoning": "Only Claim A is supported.",
}
_INCOMPLETE_FIELDS = [
    (),
    ("supported_claims",),
    ("unsupported_claims",),
    ("reasoning",),
    ("supported_claims", "unsupported_claims"),
    ("supported_claims", "reasoning"),
    ("unsupported_claims", "reasoning"),
]


def _judge(monkeypatch, route, payload):
    if route == "subscription":
        monkeypatch.setattr(auth, "subscription_available", lambda: True)

        async def query(**kwargs):
            return payload

        monkeypatch.setattr(auth, "query_subscription_structured", query)
        return FaithfulnessJudge(auth_mode="sub", model="test-model")

    block = (
        SimpleNamespace(type="tool_use", input=payload)
        if route == "tool"
        else SimpleNamespace(type="text", text=json.dumps(payload))
    )

    async def create(**kwargs):
        return SimpleNamespace(content=[block])

    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return FaithfulnessJudge(client=client, model="test-model")


@pytest.mark.parametrize("route", ["tool", "text", "subscription"])
@pytest.mark.parametrize("present_fields", _INCOMPLETE_FIELDS)
async def test_incomplete_verdict_raises_without_a_score(monkeypatch, route, present_fields):
    payload = {key: _VERDICT[key] for key in present_fields}
    judge = _judge(monkeypatch, route, payload)

    with pytest.raises(RuntimeError, match="missing required fields") as caught:
        await judge.score("Question", "A nonempty factual answer.", "Source passage.")

    for missing in set(_VERDICT) - set(present_fields):
        assert missing in str(caught.value)
    assert not _transient_primary_provider_error(caught.value)
    assert "nonempty factual answer" not in str(caught.value)
    assert "Source passage" not in str(caught.value)


@pytest.mark.parametrize("route", ["tool", "text", "subscription"])
@pytest.mark.parametrize("unsupported, expected", [(["Claim B"], 0.0), ([], 1.0)])
async def test_complete_verdict_preserves_valid_scores(monkeypatch, route, unsupported, expected):
    payload = {
        "supported_claims": [],
        "unsupported_claims": unsupported,
        "reasoning": "Unsupported claim." if unsupported else "Explicit refusal; no claims.",
    }
    judge = _judge(monkeypatch, route, payload)
    result = await judge.score("Question", "A nonempty answer.", "Source passage.")
    assert result.score == expected
    assert result.unsupported_claims == unsupported
