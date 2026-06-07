---
type: reference
name: reranker-reference
feature: reranker
depth: reference
generated_at: 2026-06-07T07:14:09.705743+00:00
source_hash: d9cc73a55820ef60156edf63a24310f219daaa440a814d281fee2195484a90ae
status: generated
---

# Reranker reference

Use `LLMReranker` to re-order keyword-retrieved candidates by relevance using Claude as a judge. Any API error falls back to the original keyword-only order, so the reranker is safe to use in production without a circuit breaker.

## Classes

| Class | Description |
|-------|-------------|
| `LLMReranker` | Re-ranks keyword retrieval candidates by relevance using Claude Haiku. |

## `LLMReranker`

```python
from attune_rag.reranker import LLMReranker
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `'claude-haiku-4-5'` | Claude model used as the relevance judge. |
| `api_key` | `str | None` | `None` | Anthropic API key. If `None`, the client reads from the environment. |
| `candidate_multiplier` | `int` | `3` | Multiplier applied to the requested result count to determine how many candidates to retrieve before re-ranking. |
| `timeout` | `float` | `60.0` | Request timeout in seconds for each Claude call. |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `rerank` | `query: str`, `hits: list[RetrievalHit]` | `list[RetrievalHit]` | Re-ranks `hits` against `query` and returns them in relevance order. Falls back to the original order if the API call fails. |

## Module constants

### `_SYSTEM`

System prompt passed to Claude on every rerank call. Instructs the model to act as a relevance judge and return a JSON array of 0-based indices ranked from most to least relevant.

| Constant | Type | Value (stem) |
|----------|------|--------------|
| `_SYSTEM` | `str` | `'You are a relevance judge for the attune-ai developer workflow documentation system…'` |

**Full value:**

```
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

Include every index exactly once. Return ONLY the JSON array — no explanation.
```

## Source files

- `src/attune_rag/reranker.py`

## Tags

`reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
