---
type: reference
name: expander-reference
feature: expander
depth: reference
generated_at: 2026-07-10T13:05:48.731840+00:00
source_hash: 8645de9f31cc8aa82ed6ad99d147639060d87eaed712a4c12d8f2c14f1355d03
status: generated
---

# Expander reference

`QueryExpander` uses Claude Haiku to broaden a query into alternative phrasings before keyword retrieval, lifting recall on queries with low surface-level overlap against target documents. Any API error falls back to the original query unchanged.

## Classes

| Class | Description |
|-------|-------------|
| `QueryExpander` | Expands a query into alternative phrasings using Claude Haiku. |

### `QueryExpander`

#### Constructor

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `model` | `str \| None` | `None` | Claude model identifier. |
| `api_key` | `str \| None` | `None` | Anthropic API key. |
| `cache` | `bool` | `True` | Whether to cache expansion results. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `expand` | `query: str` | `list[str]` | Expands a query into 3–5 alternative phrasings synchronously. |
| `expand_async` | `query: str` | `list[str]` | Expands a query into 3–5 alternative phrasings asynchronously. |

## System prompt

The `_SYSTEM` constant controls the instruction sent to Claude Haiku on every expansion call.

| Constant | Value |
|----------|-------|
| `_SYSTEM` | `"You expand developer queries for a documentation retrieval system.\nGiven a query about developer workflows and tooling, return 3-5 alternative\nphrasings as a JSON array of strings. Expose the user's actual intent:\nfeature names, tool categories, workflow synonyms, and developer jargon.\nReturn ONLY the JSON array — no explanation, no markdown fences."` |

## Source files

- `src/attune_rag/expander.py`

## Tags

`expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
