"""Subscription-first auth routing for attune-rag LLM judge calls.

Phase 2 of the ``sibling-subscription-auth`` spec (attune-ai
``docs/specs/sibling-subscription-auth/``). When attune-rag runs
under a Claude Code session (``CLAUDECODE=1``) and the
``claude-agent-sdk`` package is importable, the faithfulness
judge's LLM call routes through the user's Claude subscription via
``claude_agent_sdk.query()`` — no ``ANTHROPIC_API_KEY`` required.
Otherwise the judge falls back to the direct ``AsyncAnthropic``
path.

Mode resolution (first match wins):

1. Explicit ``auth_mode=`` argument (the CLI's ``--auth-mode``).
2. The ``ATTUNE_RAG_AUTH_MODE`` environment variable.
3. ``auto`` — subscription when detectable, else API key.

The subscription path spawns a short-lived ``claude`` CLI
subprocess per call with ``setting_sources=[]`` so user/project
settings (SessionStart hooks, CLAUDE.md context injection) never
leak into the call or pollute its stream-json channel. The judge's
guaranteed-schema contract is preserved on this path via the Agent
SDK's structured output (``output_format={"type": "json_schema",
...}`` → ``ResultMessage.structured_output``), the subscription
equivalent of the API path's forced ``tool_choice``.

Mirrors ``attune_author.auth`` (Phase 1), async-native because
``FaithfulnessJudge.score`` is async.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

#: Environment variable that pins the auth mode for every call in
#: the process. Same vocabulary as the explicit argument.
AUTH_MODE_ENV = "ATTUNE_RAG_AUTH_MODE"

VALID_AUTH_MODES = ("auto", "api", "sub")

_SK_ANT_RE = re.compile(r"sk-ant-[A-Za-z0-9_-]+")

_NO_CREDENTIALS_MESSAGE = (
    "No Anthropic credentials available: run inside a Claude Code "
    "session (subscription routing via claude-agent-sdk) or set "
    "ANTHROPIC_API_KEY."
)


class SubscriptionCallError(RuntimeError):
    """A subscription-path judge call failed (message pre-redacted)."""


def _redact(text: str) -> str:
    """Strip anything shaped like an Anthropic API key from text."""
    return _SK_ANT_RE.sub("sk-ant-<redacted>", text)


def _sdk_importable() -> bool:
    """Return True when ``claude_agent_sdk`` can be imported."""
    try:
        return importlib.util.find_spec("claude_agent_sdk") is not None
    except (ImportError, ValueError):
        return False


def subscription_available() -> bool:
    """Return True when subscription routing is possible right now.

    Requires both a detectable Claude Code session (the
    ``CLAUDECODE=1`` env var Claude Code sets in every subprocess
    it spawns) and an importable ``claude-agent-sdk``. A subscriber
    running from a plain terminal has neither and falls back to
    the API path — that's the expected, documented behavior.
    """
    return os.environ.get("CLAUDECODE") == "1" and _sdk_importable()


def api_key_available() -> bool:
    """Return True when a non-empty ``ANTHROPIC_API_KEY`` is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _requested_mode(explicit: str | None) -> str:
    """Normalize the requested mode from explicit arg or env var."""
    mode = (explicit or os.environ.get(AUTH_MODE_ENV) or "auto").strip().lower()
    if mode not in VALID_AUTH_MODES:
        raise ValueError(
            f"Invalid auth mode {mode!r}; expected one of: " + ", ".join(VALID_AUTH_MODES)
        )
    return mode


def resolve_auth_mode(explicit: str | None = None) -> str:
    """Resolve the effective auth route: ``"sub"`` or ``"api"``.

    Args:
        explicit: Explicit mode override (``auto``/``api``/``sub``).
            When ``None``, the ``ATTUNE_RAG_AUTH_MODE`` env var is
            consulted; when that's unset too, ``auto``.

    Returns:
        ``"sub"`` (subscription via the Agent SDK) or ``"api"``
        (direct Anthropic SDK).

    Raises:
        ValueError: If the mode string is invalid, or ``sub`` is
            forced while no subscription session is detectable.
    """
    mode = _requested_mode(explicit)
    if mode == "sub":
        if not subscription_available():
            raise ValueError(
                "auth mode 'sub' forced but no subscription session "
                "is detectable (CLAUDECODE is not set, or "
                "claude-agent-sdk is not installed)"
            )
        return "sub"
    if mode == "api":
        return "api"
    return "sub" if subscription_available() else "api"


def auth_telemetry() -> dict[str, float]:
    """Per-process counters of judge LLM calls by auth route.

    Stored on the function as an attribute so end-of-run summaries
    can read totals without module-level state — same idiom as
    ``attune_author.auth.auth_telemetry``. Reset via
    :func:`reset_auth_telemetry`.
    """
    state = getattr(auth_telemetry, "_state", None)
    if state is None:
        state = {"sub_calls": 0.0, "api_calls": 0.0}
        auth_telemetry._state = state  # type: ignore[attr-defined]
    return state


def reset_auth_telemetry() -> None:
    """Reset the per-process auth-route telemetry counters."""
    auth_telemetry._state = {  # type: ignore[attr-defined]
        "sub_calls": 0.0,
        "api_calls": 0.0,
    }


async def query_subscription_structured(
    *,
    system: str,
    user_message: str,
    model: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Single-turn structured completion via ``claude_agent_sdk.query()``.

    The Agent SDK's ``output_format={"type": "json_schema", ...}``
    maps to the ``claude`` CLI's ``--json-schema`` flag, so the
    returned payload is schema-validated by the CLI before it
    reaches us — the subscription-path equivalent of the API
    path's forced ``tool_choice``.

    Args:
        system: System prompt.
        user_message: User-turn content.
        model: Anthropic model ID.
        schema: JSON schema the response payload must match.

    Returns:
        The schema-validated payload dict from
        ``ResultMessage.structured_output``.

    Raises:
        SubscriptionCallError: When the stream ends without a
            structured payload (message pre-redacted).
    """
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    options = ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        # 2, not 1: with output_format the CLI spends an extra turn
        # synthesizing the schema-validated payload; max_turns=1 ends
        # the run with "Reached maximum number of turns" and no
        # structured_output (found live, 2026-06-11).
        max_turns=2,
        tools=[],  # pure completion — no tool use
        setting_sources=[],  # no user/project settings (hooks, CLAUDE.md)
        output_format={"type": "json_schema", "schema": schema},
        env={"ATTUNE_RAG_SDK_SUBPROCESS": "1"},
    )
    structured: dict[str, Any] | None = None
    async for message in query(prompt=user_message, options=options):
        if isinstance(message, ResultMessage):
            payload = message.structured_output
            if isinstance(payload, dict):
                structured = payload
    if structured is None:
        raise SubscriptionCallError(
            "subscription judge call returned no structured output "
            "(claude CLI version may predate --json-schema support)"
        )
    return structured
