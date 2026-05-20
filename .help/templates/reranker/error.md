---
type: error
name: reranker-error
feature: reranker
depth: error
generated_at: 2026-05-20T03:36:00.777378+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Reranker errors

## Common error signatures

Errors in `LLMReranker` fall into two categories: failures calling the Claude Haiku API, and failures parsing its response. When an API call fails, `LLMReranker.rerank()` falls back to the original keyword-retrieval order rather than raising — so if you're seeing no re-ranking effect, the root cause is often a silent API error rather than an exception. Errors that do propagate typically look like:

- `anthropic.APIConnectionError` or `anthropic.APITimeoutError` — the Claude API did not respond within the configured `timeout` (default `60.0` seconds).
- `anthropic.AuthenticationError` — the `api_key` passed to `LLMReranker.__init__()` is missing or invalid.
- `json.JSONDecodeError` — Claude returned a response that could not be parsed as a JSON array of indices. This can happen if the model ignored the `Return ONLY the JSON array` instruction in the system prompt.
- `IndexError` or `ValueError` — the parsed JSON array contained indices outside the range of the candidate list, or did not include every index exactly once.

## Where errors originate

All re-ranking logic runs through a single method:

- `LLMReranker.rerank(query, hits)` in `src/attune_rag/reranker.py` — sends the query and the top-K candidate documents (path + summary) to Claude Haiku, receives a ranked JSON array of 0-based indices, and returns the re-ordered `hits`. This is the only method that calls the Claude API, so every API-related error originates here.

## How to diagnose

1. **Confirm whether re-ranking is actually running.** If `rerank()` silently fell back to keyword order, the return value will match the original `hits` list. Add a log statement or assertion after calling `rerank()` to check whether the returned order differs from the input order.

2. **Check your API key and model name.** `LLMReranker` defaults to `model='claude-haiku-4-5-20251001'`. An `AuthenticationError` means the key passed via `api_key` (or inferred from the environment) is wrong or absent. A `NotFoundError` from the Anthropic SDK means the model string does not match a deployed model name.

3. **Reproduce with a minimal candidate list.** Call `rerank()` directly with a single `RetrievalHit` and a short query. If the error reproduces, capture the raw response text from the SDK before JSON parsing to see exactly what Claude returned. A response containing explanation text alongside the array means the system prompt's `Return ONLY the JSON array` constraint was not followed.

4. **Check the timeout.** If you're hitting `APITimeoutError` under normal load, construct `LLMReranker` with a larger `timeout` value. If the timeout fires consistently, check whether `candidate_multiplier` (default `3`) is producing a candidate list that's too large for Claude to rank within the budget.

5. **Validate the returned index array.** If you see `IndexError` or an unexpected re-ordering, log the raw JSON array returned by Claude and compare it against `len(hits)`. Every index from `0` to `len(hits) - 1` must appear exactly once; anything else indicates a malformed response.

## Source files

- `src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
