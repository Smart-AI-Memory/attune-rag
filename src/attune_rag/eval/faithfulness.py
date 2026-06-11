"""LLM-as-judge faithfulness scoring.

A faithful answer is one whose claims are all directly
supported by the retrieved passages. An unfaithful answer
contains hallucinations or drifts into the model's prior
knowledge.

The judge:

1. Receives the original query, the model's answer, and
   the retrieved passages.
2. Decomposes the answer into atomic factual claims.
3. Marks each claim ``supported`` or ``unsupported``
   against the passages.
4. Returns a per-answer score in ``[0, 1]``
   (supported / total).

Implemented via Anthropic tool-use with ``tool_choice``
forced to the ``report_faithfulness`` tool, so the judge's
output is always valid JSON matching a declared schema.

Auth routing (``attune_rag.auth``): under a Claude Code session
(``CLAUDECODE=1``) with ``claude-agent-sdk`` installed, the judge
call routes through the Claude subscription via the Agent SDK's
structured output — same schema guarantee, no ``ANTHROPIC_API_KEY``
required. Otherwise it uses the direct ``AsyncAnthropic`` path.
Override with ``auth_mode=`` / ``ATTUNE_RAG_AUTH_MODE``.

Extended thinking is available via ``use_thinking=True`` on
``score``. Thinking-mode forces ``tool_choice="auto"`` (an
Anthropic constraint on Claude 4 models), so the response
parser also handles the rare case where the model declines
the tool and emits a text block of JSON instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_TIMEOUT_SECONDS = 60.0
DEFAULT_THINKING_BUDGET_TOKENS = 32768


_JUDGE_SYSTEM_PROMPT = """You are a strict faithfulness judge for a retrieval-augmented \
generation (RAG) system. Your job is to decide, for each factual claim in an answer, \
whether it is directly supported by the retrieved passages.

A claim is SUPPORTED only if a passage explicitly states it. A claim is UNSUPPORTED \
if it relies on outside knowledge, reasonable inference beyond what the passages say, \
or invented details (workflow names, CLI flags, API shapes not in the passages).

Decompose the answer into atomic factual claims — one claim per verifiable assertion. \
Ignore stylistic framing, transitions, and pleasantries. If the answer is a refusal \
("the context does not cover this"), treat it as zero claims rather than a supported \
or unsupported claim.

Be strict. When in doubt, mark as UNSUPPORTED."""


_JUDGE_TOOL: dict[str, Any] = {
    "name": "report_faithfulness",
    "description": (
        "Report the faithfulness analysis of a RAG answer. Must be called "
        "exactly once with the decomposed claim list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "supported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Atomic factual claims from the answer that are "
                    "directly supported by at least one passage."
                ),
            },
            "unsupported_claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Atomic factual claims from the answer NOT supported "
                    "by the passages (hallucinations, prior-knowledge drift)."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "One short paragraph explaining the overall verdict "
                    "and any notable patterns (e.g. invented CLI flags)."
                ),
            },
        },
        "required": ["supported_claims", "unsupported_claims", "reasoning"],
    },
}


_JUDGE_USER_TEMPLATE = """### USER QUERY

{query}

### RETRIEVED PASSAGES

{passages}

### ANSWER UNDER REVIEW

{answer}

### TASK

Decompose the ANSWER into atomic factual claims. For each, decide whether the \
RETRIEVED PASSAGES directly support it. Call the `report_faithfulness` tool with \
the results."""


@dataclass(frozen=True)
class FaithfulnessResult:
    """Per-answer faithfulness verdict.

    ``score`` is ``supported / (supported + unsupported)``.
    A refusal answer (zero extracted claims) scores 1.0 —
    refusing is the correct behavior when the context
    doesn't cover the question.
    """

    score: float
    supported_claims: list[str]
    unsupported_claims: list[str]
    reasoning: str
    model: str
    thinking_used: bool = False

    @property
    def total_claims(self) -> int:
        return len(self.supported_claims) + len(self.unsupported_claims)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "total_claims": self.total_claims,
            "supported_claims": list(self.supported_claims),
            "unsupported_claims": list(self.unsupported_claims),
            "reasoning": self.reasoning,
            "model": self.model,
            "thinking_used": self.thinking_used,
        }


class FaithfulnessJudge:
    """Scores RAG answers for grounding in retrieved context.

    Requires ``attune-rag[claude]``. Uses Anthropic tool-use
    with a forced tool call so the output schema is
    guaranteed.
    """

    def __init__(
        self,
        client: AsyncAnthropic | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_JUDGE_MODEL,
        timeout: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
        auth_mode: str | None = None,
    ) -> None:
        """Construct a judge.

        Args:
            client: Injected ``AsyncAnthropic`` instance. When
                passed, ``api_key`` and ``timeout`` are ignored —
                the caller has full control of the client, and the
                judge always uses the API route (explicit injection
                wins over subscription detection).
            api_key: API key for the default client. Falls back to
                ``ANTHROPIC_API_KEY`` in the environment when None.
            model: Judge model name.
            timeout: Per-request timeout in seconds for the
                default client. Ignored when ``client`` is
                injected. A stalled network must not burn the
                benchmark loop's budget indefinitely, so the
                default is finite.
            auth_mode: ``auto``/``api``/``sub`` route override; see
                :func:`attune_rag.auth.resolve_auth_mode`. Default
                ``auto``: subscription when running under Claude
                Code with claude-agent-sdk installed, else API.

        Raises:
            ValueError: If both ``client`` and ``api_key`` are
                provided (ambiguous), the mode string is invalid,
                or ``sub`` is forced while undetectable.
            RuntimeError: If the API route is selected and the
                ``[claude]`` extra is not installed.
        """
        from attune_rag import auth as _auth

        if client is not None and api_key is not None:
            raise ValueError("Pass either client or api_key, not both — they conflict.")
        self._api_key = api_key
        self._timeout = timeout
        if client is not None:
            self._client = client
            self._route = "api"
            self._forced_sub = False
        else:
            self._route = _auth.resolve_auth_mode(auth_mode)
            self._forced_sub = _auth._requested_mode(auth_mode) == "sub"
            self._client = None
            if self._route == "api":
                self._client = self._build_api_client()
        self._model = model

    def _build_api_client(self) -> AsyncAnthropic:
        """Construct the default ``AsyncAnthropic`` client.

        Raises:
            RuntimeError: If the ``[claude]`` extra is missing.
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "FaithfulnessJudge requires the [claude] extra. "
                "Install with: pip install 'attune-rag[claude]'"
            ) from exc
        return AsyncAnthropic(api_key=self._api_key, timeout=self._timeout)

    @property
    def model(self) -> str:
        return self._model

    async def score(
        self,
        query: str,
        answer: str,
        passages: str | list[str],
        max_tokens: int = 2048,
        *,
        use_thinking: bool = False,
        thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS,
    ) -> FaithfulnessResult:
        """Score ``answer`` for faithfulness against ``passages``.

        Args:
            query: The user's original question.
            answer: The RAG system's answer.
            passages: Retrieved context — either a single
                pre-joined string or a list of passage
                strings (will be joined with separators).
            max_tokens: Budget for the judge's *reply* (the verdict
                tokens, not counting thinking). In thinking mode the
                API ``max_tokens`` field caps the combined thinking +
                response output and must exceed
                ``thinking_budget_tokens``, so we send
                ``max_tokens + thinking_budget_tokens`` to the API
                while the caller keeps the original semantic meaning.
            use_thinking: Opt into Anthropic extended thinking
                for this call. Forces ``tool_choice="auto"``
                (Anthropic constraint); parser handles both
                tool_use and text-block fallback shapes.
            thinking_budget_tokens: Ceiling for thinking
                tokens. Ignored when ``use_thinking=False``.
                Billing is per emitted token, not the budget,
                so a generous ceiling is free insurance.
        """
        if not answer or not answer.strip():
            return FaithfulnessResult(
                score=1.0,
                supported_claims=[],
                unsupported_claims=[],
                reasoning="Answer was empty; nothing to evaluate.",
                model=self._model,
                thinking_used=False,
            )

        joined = (
            passages
            if isinstance(passages, str)
            else "\n\n---\n\n".join(p.strip() for p in passages if p.strip())
        )

        user_message = _JUDGE_USER_TEMPLATE.format(
            query=query.strip(),
            passages=joined.strip() or "(no passages retrieved)",
            answer=answer.strip(),
        )

        if use_thinking:
            # Anthropic API constraint: max_tokens must be strictly
            # greater than thinking.budget_tokens, because max_tokens
            # caps the combined thinking + response output in
            # extended-thinking mode. Caller's ``max_tokens`` keeps its
            # original meaning (budget for the judge's reply); we add
            # thinking_budget on top before sending.
            effective_max_tokens = max_tokens + thinking_budget_tokens
        else:
            effective_max_tokens = max_tokens

        from attune_rag import auth as _auth

        payload: dict[str, Any] | None = None
        if self._route == "sub":
            if use_thinking:
                logger.warning(
                    "use_thinking is not supported on the subscription "
                    "route; scoring without extended thinking."
                )
            try:
                payload = await _auth.query_subscription_structured(
                    system=_JUDGE_SYSTEM_PROMPT,
                    user_message=user_message,
                    model=self._model,
                    schema=_JUDGE_TOOL["input_schema"],
                )
            except Exception as exc:  # noqa: BLE001
                # INTENTIONAL: every subscription-path failure funnels
                # through one redaction + fallback decision point,
                # mirroring attune_author.auth.call_llm. `from None`
                # keeps unredacted text out of __cause__.
                if self._forced_sub or not _auth.api_key_available():
                    raise RuntimeError(_auth._redact(str(exc))) from None
                logger.warning(
                    "Subscription judge call failed; falling back to " "the API key path: %s",
                    _auth._redact(str(exc)),
                )
            else:
                _auth.auth_telemetry()["sub_calls"] += 1

        via_api = payload is None
        if via_api:
            if self._client is None:
                self._client = self._build_api_client()
            request_kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": effective_max_tokens,
                "system": _JUDGE_SYSTEM_PROMPT,
                "tools": [_JUDGE_TOOL],
                "messages": [{"role": "user", "content": user_message}],
            }
            if use_thinking:
                # Anthropic constraint on Claude 4: thinking + tools
                # requires tool_choice in {"auto", "none"}. Forced
                # tool_choice is incompatible.
                request_kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_budget_tokens,
                }
                request_kwargs["tool_choice"] = {"type": "auto"}
            else:
                request_kwargs["tool_choice"] = {
                    "type": "tool",
                    "name": "report_faithfulness",
                }

            response = await self._client.messages.create(**request_kwargs)

            payload = _extract_judge_payload(response)
            _auth.auth_telemetry()["api_calls"] += 1
        supported, unsupported, reasoning = _parse_judge_payload(payload)

        total = len(supported) + len(unsupported)
        # Refusal (zero claims) is a faithful outcome.
        score = 1.0 if total == 0 else len(supported) / total

        return FaithfulnessResult(
            score=score,
            supported_claims=supported,
            unsupported_claims=unsupported,
            reasoning=reasoning,
            model=self._model,
            thinking_used=use_thinking and via_api,
        )


def _parse_judge_payload(payload: dict[str, Any]) -> tuple[list[str], list[str], str]:
    """Validate and extract fields from the judge's tool_use input.

    ``tool_choice`` forces the schema, but a future SDK / API
    change could still hand us an unexpected shape. Validate
    each field explicitly so the error points at the schema
    violation instead of surfacing as a cryptic ``TypeError``
    on ``len()`` further down.
    """

    def _as_str_list(value: Any, field_name: str) -> list[str]:
        if not isinstance(value, list):
            raise RuntimeError(
                f"FaithfulnessJudge: expected list for "
                f"{field_name!r}, got {type(value).__name__}"
            )
        return [str(item) for item in value]

    supported = _as_str_list(payload.get("supported_claims", []), "supported_claims")
    unsupported = _as_str_list(payload.get("unsupported_claims", []), "unsupported_claims")

    reasoning_raw = payload.get("reasoning", "")
    if not isinstance(reasoning_raw, str):
        raise RuntimeError(
            f"FaithfulnessJudge: expected str for 'reasoning', "
            f"got {type(reasoning_raw).__name__}"
        )

    return supported, unsupported, reasoning_raw.strip()


def _strip_code_fences(text: str) -> str:
    """Strip a leading ```json (or ```) fence and trailing ``` if present.

    Thinking-mode responses occasionally wrap the JSON payload
    in a fenced code block. Stripping is conservative — only
    when both opening and closing fences are detected.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2 or not lines[-1].rstrip().endswith("```"):
        return stripped
    # Drop first line (```json or ```) and last line (```).
    return "\n".join(lines[1:-1]).strip()


def _extract_judge_payload(response: Any) -> dict[str, Any]:
    """Pull the structured judge payload from an Anthropic response.

    Walks ``response.content`` block by block:

    1. ``tool_use`` block → return its ``input`` dict (the
       schema-guaranteed happy path; identical to pre-thinking
       behavior).
    2. ``text`` block (only if no ``tool_use`` was found) →
       JSON-parse the text. Covers the rare "thinking-enabled
       model declined to call the tool" case.
    3. ``thinking`` blocks are skipped — they carry reasoning,
       not the verdict.

    Raises ``RuntimeError`` with a diagnostic snippet when
    neither path yields a dict.
    """
    text_block_payload: str | None = None
    for block in getattr(response, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
        elif btype == "text" and text_block_payload is None:
            text_block_payload = getattr(block, "text", None)
        # thinking blocks: skip

    if text_block_payload:
        try:
            parsed = json.loads(_strip_code_fences(text_block_payload))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "FaithfulnessJudge: model declined tool_use and returned "
                "unparseable text. First 200 chars: "
                f"{text_block_payload[:200]!r}"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise RuntimeError(
            "FaithfulnessJudge: text-block JSON was not an object "
            f"(got {type(parsed).__name__})."
        )

    raise RuntimeError(
        "FaithfulnessJudge: response contained no tool_use or text block. "
        "Check API version or tool_choice support."
    )
