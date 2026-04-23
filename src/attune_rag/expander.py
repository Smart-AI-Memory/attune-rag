"""QueryExpander — LLM-powered query expansion for improved retrieval recall."""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_SYSTEM = """\
You expand developer queries for a documentation retrieval system.
Given a query about developer workflows and tooling, return 3-5 alternative
phrasings as a JSON array of strings. Expose the user's actual intent:
feature names, tool categories, workflow synonyms, and developer jargon.
Return ONLY the JSON array — no explanation, no markdown fences."""


class QueryExpander:
    """Uses Claude Haiku to expand a query into alternative phrasings.

    Each expansion is joined with the original query before tokenisation so
    the keyword retriever sees a richer token set without any code changes to
    the retriever itself.  Requires the ``[claude]`` extra.

    >>> expander = QueryExpander()
    >>> expander.expand("publish to PyPI")
    ['release preparation', 'package publishing workflow', ...]
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        cache: bool = True,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._cache: dict[str, list[str]] | None = {} if cache else None
        self._client = None

    @property
    def _anthropic(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "QueryExpander requires the [claude] extra. "
                    "Install with: pip install 'attune-rag[claude]'"
                ) from exc
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def expand(self, query: str) -> list[str]:
        """Return alternative phrasings for *query*.

        Returns an empty list on any API or parse error so the caller
        always falls back gracefully to keyword-only retrieval.
        """
        if self._cache is not None and query in self._cache:
            return self._cache[query]

        try:
            response = self._anthropic.messages.create(
                model=self._model,
                max_tokens=200,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": query}],
            )
            raw = response.content[0].text.strip()
            expansions = json.loads(raw)
            if not isinstance(expansions, list):
                expansions = []
            expansions = [str(e) for e in expansions if e]
        except Exception as exc:  # noqa: BLE001
            logger.debug("QueryExpander.expand failed: %s", exc)
            expansions = []

        if self._cache is not None:
            self._cache[query] = expansions
        return expansions
