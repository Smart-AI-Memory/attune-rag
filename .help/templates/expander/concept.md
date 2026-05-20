---
type: concept
name: expander-concept
feature: expander
depth: concept
generated_at: 2026-05-20T02:45:15.467956+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Expander

`QueryExpander` is a retrieval preprocessing step that uses Claude Haiku to rewrite a single user query into 3–5 alternative phrasings, improving the chances that at least one phrasing matches the surface form of relevant documents.

## The problem it solves

Keyword and vector retrieval both depend on surface-level overlap between the query and the indexed content. When a user asks "how do I roll back a deploy?" but the documentation says "revert a release," a direct lookup may return nothing useful. Query expansion bridges that gap by generating synonyms, feature names, tool-category terms, and developer jargon before the retrieval step runs.

## How query expansion works

When you call `expand()` or `expand_async()`, `QueryExpander` sends the original query to Claude Haiku with a fixed system prompt. That prompt instructs the model to return **only** a JSON array of 3–5 alternative phrasings — no prose, no markdown fences. The caller receives the array of strings and can feed each phrasing into the retrieval pipeline independently.

If the Claude API call fails for any reason, `QueryExpander` falls back to the original query, so retrieval still runs rather than erroring out.

Caching is enabled by default. Repeated calls with the same query string return the cached expansion without a second API round-trip.

## The `QueryExpander` class

`QueryExpander` is the single public interface in this module.

| Member | Signature | What it does |
|---|---|---|
| `__init__` | `(model='claude-haiku-4-5-20251001', api_key=None, cache=True)` | Configures the model, optional API key, and whether to cache expansions |
| `expand` | `(query: str) -> list[str]` | Synchronously returns a list of alternative phrasings |
| `expand_async` | `(query: str) -> list[str]` | Asynchronous equivalent of `expand` |

The default model is `claude-haiku-4-5-20251001`. You can substitute a different model string at construction time if your deployment uses a different Claude variant.

## When expansion matters most

Query expansion has the highest impact when:

- The user's vocabulary differs from the documentation's vocabulary (for example, a user asking about "feature flags" when docs say "toggles")
- Queries are short or ambiguous, giving retrieval little signal to work with
- The indexed corpus uses precise technical terminology that users are unlikely to reproduce exactly

For queries that already use exact identifiers — class names, CLI flags, error codes — expansion adds less value, but the cache prevents it from adding latency on repeated lookups.
