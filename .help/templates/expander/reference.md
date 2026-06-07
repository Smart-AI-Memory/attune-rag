---
type: reference
name: expander-reference
feature: expander
depth: reference
generated_at: 2026-06-07T07:13:53.302593+00:00
source_hash: fee9ea3e96d976b16a96673b646ba25f945f41ad6136204efbd13aaa334ccf76
status: generated
---

# Expander reference

Use `QueryExpander` to broaden a user query into alternative phrasings before retrieval. The expander calls Claude Haiku to generate 3–5 rephrased versions of the original query, improving recall when the query has low surface-level overlap with target documents. Any API error falls back to the original query unchanged.

## Classes

| Class | Description |
|-------|-------------|
| `QueryExpander` | Expands a query into alternative phrasings using Claude Haiku. |

### `QueryExpander`

```python
from attune_rag.expander import QueryExpander
```

#### Constructor

| Parameters | Type | Default |
|------------|------|---------|
| `model` | `str` | `'claude-haiku-4-5'` |
| `api_key` | `str | None` | `None` |
| `cache` | `bool` | `True` |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `expand` | `query: str` | `list[str]` | Expands a query into alternative phrasings synchronously. |
| `expand_async` | `query: str` | `list[str]` | Expands a query into alternative phrasings asynchronously. |

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `_SYSTEM` | `str` | `"You expand developer queries for a documentation retrieval system.\nGiven a query about developer workflows and tooling, return 3-5 alternative\nphrasings as a JSON array of strings. Expose the user's actual intent:\nfeature names, tool categories, workflow synonyms, and developer jargon.\nReturn ONLY the JSON array — no explanation, no markdown fences."` |

## Source files

- `src/attune_rag/expander.py`

## Tags

`expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
