---
type: concept
name: expander-concept
feature: expander
depth: concept
generated_at: 2026-06-07T07:13:53.291643+00:00
source_hash: fee9ea3e96d976b16a96673b646ba25f945f41ad6136204efbd13aaa334ccf76
status: generated
---

# Expander

`QueryExpander` is a retrieval utility that rewrites a single user query into 3–5 alternative phrasings using Claude Haiku, so that keyword search can match documents even when the user's wording doesn't overlap with the indexed text.

## The retrieval recall problem

Keyword-based retrieval fails when the user's phrasing doesn't share surface-level tokens with the target document. A developer asking "how do I silence a linter warning?" will miss documents indexed under "suppressing diagnostic rules" or "disabling checks." Query expansion closes this gap by generating synonymous phrasings before the retrieval step runs.

## How `QueryExpander` works

When you call `expand(query)` or `expand_async(query)`, `QueryExpander` sends the query to Claude Haiku with a system prompt that instructs the model to expose the user's underlying intent — feature names, tool categories, workflow synonyms, and developer jargon — and return the results as a JSON array of strings. Your retrieval layer then fans out across all returned phrasings instead of the original query alone.

If the API call fails for any reason, `QueryExpander` falls back to the original query, so a network hiccup or quota error never blocks retrieval entirely.

Caching is on by default (`cache=True`). Repeated calls with the same query string reuse the previously generated expansions rather than making a second API call.

## Construction and defaults

```python
from attune_rag.expander import QueryExpander

expander = QueryExpander(
    model="claude-haiku-4-5",  # default
    api_key=None,              # falls back to environment credential
    cache=True,                # deduplicates identical queries
)
```

| Parameter | Default | Effect |
|-----------|---------|--------|
| `model` | `'claude-haiku-4-5'` | Which Claude model generates the expansions |
| `api_key` | `None` | Explicit API key; omit to use the environment credential |
| `cache` | `True` | Caches results per unique query string |

## Sync and async interfaces

`QueryExpander` exposes two methods with identical semantics:

- **`expand(query: str) -> list[str]`** — blocking call; use this in synchronous retrieval pipelines.
- **`expand_async(query: str) -> list[str]`** — non-blocking call; use this when your retrieval layer is already async.

Both return a list of alternative query strings. Pass each string to your retrieval backend and merge the results before ranking.

## When query expansion matters

Query expansion has the most impact when:

- Your document corpus uses technical jargon or product-specific terminology that users are unlikely to reproduce verbatim.
- You're running BM25 or another token-overlap retrieval strategy with no semantic embedding fallback.
- Query volume is low enough that the added latency and API cost per call is acceptable.

If your retrieval pipeline already uses dense vector search with broad semantic coverage, the incremental recall gain from expansion is smaller and may not justify the extra API call.
