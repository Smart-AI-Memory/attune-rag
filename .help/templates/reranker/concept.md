---
type: concept
name: reranker-concept
feature: reranker
depth: concept
generated_at: 2026-06-07T07:14:09.690679+00:00
source_hash: d9cc73a55820ef60156edf63a24310f219daaa440a814d281fee2195484a90ae
status: generated
---

# Reranker

`LLMReranker` is a Claude-powered relevance judge that reorders keyword-retrieval candidates so the most relevant documents rise to the top before your application consumes the results.

## How it fits into retrieval

Keyword retrieval is fast but scores documents by term overlap, not intent. A query like "fix tests" can surface a CI/CD setup guide above an actual fix tool simply because both documents share the word "tests."

`LLMReranker` addresses this by adding a second pass. After your retrieval stage returns a candidate list, `LLMReranker.rerank(query, hits)` sends those candidates to `claude-haiku-4-5` acting as a relevance judge. The model returns a ranked ordering of the candidates by semantic fit, and `rerank` gives you back the same `RetrievalHit` objects in that improved order.

The `candidate_multiplier` parameter (default `3`) controls how many candidates your retrieval stage should fetch relative to the number of results you ultimately want. For example, if you want 5 final results, retrieve 15 candidates and let the reranker select the best ordering. Fetching a wider pool gives the judge more signal to work with.

## Failure behavior

`LLMReranker` is opt-in and fail-safe. If the Claude API call times out (configurable via `timeout`, default `60.0` seconds) or returns an error, `rerank` falls back to the original keyword-retrieved order. Your pipeline keeps running; it just skips the reranking step.

## Relevant judgment criteria

The system prompt baked into `LLMReranker` encodes domain-specific ranking guidance. For example:

- For release or publish queries, canonical workflow documents are preferred over task guides.
- For "fix tests" or "CI pipeline failing" queries, fix-oriented documents are ranked above setup guides.
- Every candidate index appears exactly once in the output — the reranker reorders, never drops results.

This means the quality of reranking depends partly on how well your document paths and summaries signal their purpose to the model.

## Key surface

| Name | Role |
|---|---|
| `LLMReranker(model, api_key, candidate_multiplier, timeout)` | Instantiates the reranker. `model` defaults to `'claude-haiku-4-5'`; pass `api_key` if you are not using an environment variable. |
| `rerank(query, hits)` | Takes a query string and a list of `RetrievalHit` objects; returns the same list reordered by relevance. |
