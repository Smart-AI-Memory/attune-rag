"""LLMReranker — Claude-powered re-ranking for improved retrieval precision."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import RetrievalHit

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a relevance judge for the attune-ai developer workflow documentation system.
Given a user query and a numbered list of candidate documents (path + summary),
return a JSON array of 0-based indices ranking candidates from most to least relevant.

Ranking guidance:
- Paths with "tool-" (e.g. concepts/tool-release-prep.md) are canonical attune workflow
  references. Paths with "skill-", "task-", or "use-" are quickstarts and task guides.
  For workflow-goal queries, rank tool-* docs above skill-*, task-*, and use-* docs.
- "Version bump", "changelog", "release", "publish", or "ship" → prefer tool-release-prep.md
  (full release workflow with health, security, and changelog checks).
- "CI pipeline failing", "tests failing", "fix tests" → prefer tool-fix-test.md or
  skill-fix-test.md over task-ci-cd-pipeline.md (a setup guide, not a fix tool).
- "Orchestrate", "coordinate", or "manage" documentation → prefer tool-doc-orchestrator.md.
- "Publish to PyPI" → prefer tool-release-prep.md over task-package-publishing.md.

Include every index exactly once. Return ONLY the JSON array — no explanation."""


class LLMReranker:
    """Uses Claude Haiku to re-rank keyword retrieval candidates by relevance.

    The pipeline retrieves a wider candidate set (k * candidate_multiplier)
    and passes it to the reranker, which returns the top-k results in
    relevance order.  Requires the ``[claude]`` extra.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
        candidate_multiplier: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self.candidate_multiplier = candidate_multiplier
        self._client = None

    @property
    def _anthropic(self):
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "LLMReranker requires the [claude] extra. "
                    "Install with: pip install 'attune-rag[claude]'"
                ) from exc
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def rerank(self, query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
        """Re-rank *hits* by relevance to *query*.

        Returns the original order unchanged if re-ranking fails so the
        caller always receives a valid result.
        """
        if len(hits) <= 1:
            return hits

        lines = []
        for i, hit in enumerate(hits):
            summary = hit.entry.summary or ""
            lines.append(f"[{i}] {hit.entry.path}: {summary[:120]}")
        candidates_text = "\n".join(lines)

        prompt = (
            f"Query: {query}\n\nCandidates:\n{candidates_text}\n\n"
            "Return a JSON array of indices, most relevant first."
        )

        try:
            response = self._anthropic.messages.create(
                model=self._model,
                max_tokens=100,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            indices = json.loads(raw)
            if not isinstance(indices, list):
                return hits

            seen: set[int] = set()
            reranked: list[RetrievalHit] = []
            for i in indices:
                if isinstance(i, int) and 0 <= i < len(hits) and i not in seen:
                    reranked.append(hits[i])
                    seen.add(i)
            for i, hit in enumerate(hits):
                if i not in seen:
                    reranked.append(hit)
            return reranked

        except Exception as exc:  # noqa: BLE001
            logger.debug("LLMReranker.rerank failed: %s", exc)
            return hits
