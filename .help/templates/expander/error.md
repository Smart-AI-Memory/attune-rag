---
type: error
name: expander-error
feature: expander
depth: error
generated_at: 2026-05-20T03:34:53.265260+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Expander errors

## Common error signatures

Errors from `QueryExpander` fall into three categories: API connectivity failures, malformed LLM responses, and authentication problems.

- **Anthropic API errors** — network timeouts, rate limits, or HTTP errors raised when `expand()` or `expand_async()` calls Claude Haiku. These typically appear as SDK-level exceptions such as `anthropic.APIConnectionError`, `anthropic.RateLimitError`, or `anthropic.APIStatusError`.
- **JSON parse failures** — `json.JSONDecodeError` raised when the model returns a response that isn't a valid JSON array. The system prompt instructs the model to return *only* a JSON array of strings; any deviation (markdown fences, explanatory text, empty response) causes this failure.
- **Authentication errors** — `anthropic.AuthenticationError` when the `api_key` argument is `None` and no `ANTHROPIC_API_KEY` environment variable is set, or when the provided key is invalid.

## Where errors originate

All errors originate in `src/attune_rag/expander.py`. The two entry points are:

- `QueryExpander.expand(query)` — synchronous path; raises synchronously on API or parse failure.
- `QueryExpander.expand_async(query)` — async path; the same failure modes apply, but exceptions are raised inside the coroutine and must be awaited to surface.

Both methods share the same LLM call and JSON parsing logic, so a failure in one likely reproduces in the other.

## How to diagnose

1. **Check the exception type first.** The type tells you which layer failed:
   - An `anthropic.AuthenticationError` means the API key is missing or wrong — verify the `api_key` argument passed to `QueryExpander.__init__` or the `ANTHROPIC_API_KEY` environment variable.
   - An `anthropic.RateLimitError` or `anthropic.APIConnectionError` means the Anthropic API was unreachable or throttled — retry with backoff or check your account limits.
   - A `json.JSONDecodeError` means the model returned something other than a plain JSON array. Log the raw response text to see what the model actually returned.

2. **Inspect the raw model response.** If you see a `json.JSONDecodeError`, the model ignored the system prompt constraint ("Return ONLY the JSON array — no explanation, no markdown fences"). Capture the `content` field of the API response before parsing to confirm what text triggered the failure.

3. **Verify caching behavior.** `QueryExpander` accepts a `cache=True` parameter. If you are seeing stale or unexpected expansions rather than an exception, confirm whether caching is returning a previously stored result instead of making a fresh API call.

4. **Confirm the model identifier.** The default model is `claude-haiku-4-5-20251001`. If you pass a custom `model` string to `__init__` and receive an `anthropic.NotFoundError` or similar, verify the model name is correct and available on your account.

## Source files

- `src/attune_rag/expander.py`

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
