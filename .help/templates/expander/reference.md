---
type: reference
name: expander-reference
feature: expander
depth: reference
generated_at: 2026-05-20T02:45:15.482731+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Expander reference

Use `QueryExpander` to broaden a user query into alternative phrasings before retrieval. The class calls Claude Haiku to generate 3–5 synonyms and rephrasing variants, lifting recall on queries with low surface-level overlap against target documents. Any API error falls back to the original query unchanged.

## Classes

| Class | Description |
|-------|-------------|
| `QueryExpander` | Expands a query into alternative phrasings using Claude Haiku. |

## `QueryExpander`

### Constructor

| Parameters | Type | Default |
|------------|------|---------|
| `model` | `str` | `'claude-haiku-4-5-20251001'` |
| `api_key` | `str | None` | `None` |
| `cache` | `bool` | `True` |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `expand` | `query: str` | `list[str]` | Expands a query synchronously into alternative phrasings. |
| `expand_async` | `query: str` | `list[str]` | Expands a query asynchronously into alternative phrasings. |

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `_SYSTEM` | `str` | `"You expand developer queries for a documentation retrieval system.\nGiven a query about developer workflows and tooling, return 3-5 alternative\nphrasings as a JSON array of strings. Expose the user's actual intent:\nfeature names, tool categories, workflow synonyms, and developer jargon.\nReturn ONLY the JSON array — no explanation, no markdown fences."` |

## Source files

- `src/attune_rag/expander.py`

## Tags

`expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
