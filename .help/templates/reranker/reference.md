---
type: reference
name: reranker-reference
feature: reranker
depth: reference
generated_at: 2026-07-10T13:06:04.032180+00:00
source_hash: c828b6c3ccd4f66d997d42c41fc386540dc0978c13f780db0e2920cdbb911f6d
status: generated
---

# Reranker reference

Use `LLMReranker` to re-rank keyword retrieval candidates with Claude Haiku as a relevance judge, improving retrieval precision over keyword-only ordering. Any API error falls back to the original keyword order, so opt-in use is safe by default.

## Classes

| Class | Description |
|-------|-------------|
| `LLMReranker` | Uses Claude Haiku to re-rank keyword retrieval candidates by relevance. |

### `LLMReranker`

#### Constructor

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `model` | `str \| None` | `None` | Claude model identifier to use for re-ranking. |
| `api_key` | `str \| None` | `None` | Anthropic API key. If `None`, uses the environment default. |
| `candidate_multiplier` | `int` | `3` | Multiplier applied to the requested result count to determine the candidate pool size passed to the LLM. |
| `timeout` | `float` | `60.0` | Request timeout in seconds. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `rerank` | `query: str`, `hits: list[RetrievalHit]` | `list[RetrievalHit]` | Re-ranks `hits` against `query` using Claude Haiku and returns candidates ordered from most to least relevant. |

## Constants

### `_SYSTEM`

System prompt passed to Claude as the relevance judge. Instructs the model to return a JSON array of 0-based indices ranking candidates from most to least relevant, with no additional explanation.

| Constant | Type | Value |
|----------|------|-------|
| `_SYSTEM` | `str` | `'You are a relevance judge for the attune-ai developer workflow documentation system.\nGiven a user query and a numbered list of candidate documents (path + summary),\nreturn a JSON array of 0-based indices ranking candidates from most to least relevant.\n\nRanking guidance:\n- Paths with "tool-" (e.g. concepts/tool-release-prep.md) are canonical attune workflow\n  references. Paths with "skill-", "task-", or "use-" are quickstarts and task guides.\n  For workflow-goal queries, rank tool-* docs above skill-*, task-*, and use-* docs.\n- "Version bump", "changelog", "release", "publish", or "ship" → prefer tool-release-prep.md\n  (full release workflow with health, security, and changelog checks).\n- "CI pipeline failing", "tests failing", "fix tests" → prefer tool-fix-test.md or\n  skill-fix-test.md over task-ci-cd-pipeline.md (a setup guide, not a fix tool).\n- "Orchestrate", "coordinate", or "manage" documentation → prefer tool-doc-orchestrator.md.\n- "Publish to PyPI" → prefer tool-release-prep.md over task-package-publishing.md.\n\nInclude every index exactly once. Return ONLY the JSON array — no explanation.'` |

## Source files

- `src/attune_rag/reranker.py`

## Tags

`reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
