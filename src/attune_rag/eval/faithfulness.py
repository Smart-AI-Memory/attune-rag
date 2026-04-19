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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic


DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_JUDGE_TIMEOUT_SECONDS = 60.0


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
    ) -> None:
        """Construct a judge.

        Args:
            client: Injected ``AsyncAnthropic`` instance. When
                passed, ``api_key`` and ``timeout`` are ignored —
                the caller has full control of the client.
            api_key: API key for the default client. Falls back to
                ``ANTHROPIC_API_KEY`` in the environment when None.
            model: Judge model name.
            timeout: Per-request timeout in seconds for the
                default client. Ignored when ``client`` is
                injected. A stalled network must not burn the
                benchmark loop's budget indefinitely, so the
                default is finite.

        Raises:
            ValueError: If both ``client`` and ``api_key`` are
                provided (ambiguous).
            RuntimeError: If ``client`` is None and the ``[claude]``
                extra is not installed.
        """
        if client is not None and api_key is not None:
            raise ValueError("Pass either client or api_key, not both — they conflict.")
        if client is not None:
            self._client = client
        else:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as exc:
                raise RuntimeError(
                    "FaithfulnessJudge requires the [claude] extra. "
                    "Install with: pip install 'attune-rag[claude]'"
                ) from exc
            self._client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def score(
        self,
        query: str,
        answer: str,
        passages: str | list[str],
        max_tokens: int = 2048,
    ) -> FaithfulnessResult:
        """Score ``answer`` for faithfulness against ``passages``.

        Args:
            query: The user's original question.
            answer: The RAG system's answer.
            passages: Retrieved context — either a single
                pre-joined string or a list of passage
                strings (will be joined with separators).
            max_tokens: Budget for the judge's reply.
        """
        if not answer or not answer.strip():
            return FaithfulnessResult(
                score=1.0,
                supported_claims=[],
                unsupported_claims=[],
                reasoning="Answer was empty; nothing to evaluate.",
                model=self._model,
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

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=_JUDGE_SYSTEM_PROMPT,
            tools=[_JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "report_faithfulness"},
            messages=[{"role": "user", "content": user_message}],
        )

        payload = _extract_tool_input(response)
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


def _extract_tool_input(response: Any) -> dict[str, Any]:
    """Pull the forced tool-use block out of an Anthropic response."""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use":
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
    raise RuntimeError(
        "FaithfulnessJudge: Claude did not emit a tool_use block for "
        "report_faithfulness. Check API version or tool_choice support."
    )
