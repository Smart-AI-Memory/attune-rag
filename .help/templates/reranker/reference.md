---
type: reference
name: reranker-reference
feature: reranker
depth: reference
generated_at: 2026-05-20T03:36:00.773254+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Reranker reference

Use `LLMReranker` to improve retrieval precision by re-ranking keyword-retrieved candidates with Claude Haiku as a relevance judge. The reranker is opt-in and fail-safe — any API error falls back to the original keyword-only order.

## Classes

| Class | Description |
|-------|-------------|
| `LLMReranker` | Uses Claude Haiku to re-rank keyword retrieval candidates by relevance. |

### `LLMReranker`

#### Constructor

| Parameters | Type | Default |
|------------|------|---------|
| `model` | `str` | `'claude-haiku-4-5-20251001'` |
| `api_key` | `str | None` | `None` |
| `candidate_multiplier` | `int` | `3` |
| `timeout` | `float` | `60.0` |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `rerank` | `query: str`, `hits: list[RetrievalHit]` | `list[RetrievalHit]` | Re-ranks retrieval hits by relevance to the query. |

## System prompt

The relevance-ranking prompt sent to Claude Haiku instructs the model to return a JSON array of 0-based indices ordering candidates from most to least relevant. The full prompt text is:

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
