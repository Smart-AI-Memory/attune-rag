---
type: reference
name: expander-reference
feature: expander
depth: reference
generated_at: 2026-05-20T03:34:53.261428+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Expander reference

Use `QueryExpander` to expand a user query into alternative phrasings before retrieval. The class calls Claude Haiku to generate 3–5 rewordings that surface feature names, tool categories, workflow synonyms, and developer jargon, improving recall when the original query has low surface-level overlap with target documents.

## Classes

| Class | Description |
|-------|-------------|
| `QueryExpander` | Uses Claude Haiku to expand a query into alternative phrasings. |

### `QueryExpander`

#### Constructor

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `model` | `str` | `'claude-haiku-4-5-20251001'` | Claude model to use for expansion. |
| `api_key` | `str \| None` | `None` | Anthropic API key. If `None`, falls back to the environment variable. |
| `cache` | `bool` | `True` | Whether to cache expansion results. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `expand` | `query: str` | `list[str]` | Synchronously expands a query into alternative phrasings. |
| `expand_async` | `query: str` | `list[str]` | Asynchronously expands a query into alternative phrasings. |

## System prompt

The `_SYSTEM` constant controls the instruction sent to Claude Haiku on every expansion request.

| Constant | Value |
|----------|-------|
| `_SYSTEM` | `"You expand developer queries for a documentation retrieval system.\nGiven a query about developer workflows and tooling, return 3-5 alternative\nphrasings as a JSON array of strings. Expose the user's actual intent:\nfeature names, tool categories, workflow synonyms, and developer jargon.\nReturn ONLY the JSON array — no explanation, no markdown fences."` |

## Source files

- `src/attune_rag/expander.py`

## Tags

`expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
