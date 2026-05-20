---
type: concept
name: expander-concept
feature: expander
depth: concept
generated_at: 2026-05-20T03:34:53.251463+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Query Expander

`QueryExpander` is an LLM-powered component that rewrites a single user query into 3–5 alternative phrasings, improving retrieval recall when the user's wording differs from the terminology used in your documentation.

## The problem it solves

Keyword-based retrieval fails when the user's vocabulary doesn't match the documents. For example, a developer asking "how do I set up auth" might miss documents that use "authentication configuration" or "OAuth integration." `QueryExpander` bridges this gap by generating synonyms, feature names, tool categories, and developer jargon that expose the user's actual intent before the retrieval step runs.

## How query expansion works

When you call `expand()` or `expand_async()` on a query string, `QueryExpander` sends the query to Claude Haiku with a system prompt that instructs the model to return a JSON array of alternative phrasings — nothing else, no explanation or markdown. The result is a list of strings you can pass to your retrieval pipeline alongside or instead of the original query.

For instance, a query like `"CI pipeline caching"` might expand to:

```json
["CI cache configuration", "build cache in GitHub Actions", "pipeline dependency caching", "cache layers in CI/CD", "speed up CI with caching"]
```

Each alternative targets a different surface form of the same underlying intent.

## Component overview

`QueryExpander` (defined in `src/attune_rag/expander.py`) is the single class in this module. Its constructor accepts three parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `model` | `claude-haiku-4-5-20251001` | The Claude model used for expansion |
| `api_key` | `None` | Anthropic API key; falls back to environment variable if omitted |
| `cache` | `True` | Caches expansion results to avoid redundant API calls for repeated queries |

Both `expand()` and `expand_async()` accept a query string and return `list[str]`. Use `expand_async()` in async retrieval pipelines to avoid blocking.

## Failure behavior

`QueryExpander` is designed to be fail-safe. If the API call fails or the model returns malformed output, the expander falls back gracefully so your retrieval pipeline continues with the original query rather than raising an exception.

## When query expansion matters

Query expansion is most valuable when:

- Your documentation uses precise technical terminology that users are unlikely to reproduce verbatim.
- You are running hybrid or semantic retrieval and want to increase candidate recall before a reranking step.
- Users are exploring unfamiliar tooling and may not yet know the canonical names for features they need.

It adds latency and an API call for each unique query, so consider whether the recall improvement justifies the cost for your use case, particularly if your queries are already highly structured.
