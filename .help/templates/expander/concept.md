---
type: concept
name: expander-concept
feature: expander
depth: concept
generated_at: 2026-07-10T13:05:48.719025+00:00
source_hash: 8645de9f31cc8aa82ed6ad99d147639060d87eaed712a4c12d8f2c14f1355d03
status: generated
---

# Expander

`QueryExpander` is a retrieval preprocessing step that uses Claude Haiku to rewrite a single user query into 3–5 alternative phrasings, so that keyword search can match documents even when the user's wording doesn't overlap with the words in those documents.

## The problem it solves

Keyword and hybrid retrieval systems match on surface-level terms. If a user asks "how do I roll back a deployment?" but the relevant document talks about "reverting a release," a lexical search misses it entirely. Query expansion bridges that gap by generating synonyms, related feature names, tool categories, and developer jargon before the search runs — without requiring the user to know the right terminology in advance.

## How query expansion works

When you call `expand()` or `expand_async()`, `QueryExpander` sends your query to Claude Haiku with a fixed system prompt that instructs the model to return a JSON array of 3–5 alternative phrasings. The model is told to surface the user's actual intent through feature names, tool categories, workflow synonyms, and developer jargon. The expanded terms are then passed downstream to the retrieval layer alongside — or instead of — the original query.

A concrete example: a query like `"template caching"` might expand to phrasings like `"memoizing rendered templates"`, `"avoiding redundant doc generation"`, and `"template output reuse"`, each of which has a better chance of matching a different document.

If the Claude API call fails for any reason, `QueryExpander` falls back to the original query, so retrieval continues rather than halting.

## Key design decisions

| Decision | Detail |
|---|---|
| **Model** | Claude Haiku by default; overridable via the `model` parameter at construction time |
| **API key** | Passed at construction via `api_key`; if omitted, falls back to the environment |
| **Response format** | The system prompt instructs the model to return only a raw JSON array — no markdown fences, no explanation — making parsing deterministic |
| **Caching** | Enabled by default (`cache=True`); identical queries return cached expansions without an additional API round-trip |
| **Async support** | `expand_async()` mirrors `expand()` for use in async retrieval pipelines |

## When query expansion matters

Query expansion has the most impact when:

- Queries use informal or abbreviated language that doesn't match documentation vocabulary
- The corpus uses domain-specific jargon that users wouldn't naturally reach for
- Hybrid retrieval is already in place and you want to lift recall without reindexing

It has less impact when queries are already precise and match document terms directly, since the expanded phrasings add latency (one LLM call per query) without changing which documents rank highest.
