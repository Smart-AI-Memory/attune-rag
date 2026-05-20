---
type: troubleshooting
name: expander-troubleshooting
feature: expander
depth: troubleshooting
generated_at: 2026-05-20T03:34:53.272817+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Troubleshoot expander

## Before you start

`QueryExpander` uses Claude Haiku to rewrite a user query into 3–5 alternative phrasings before retrieval, improving recall when the original query has low surface-level overlap with your documents. It is opt-in and fail-safe: if the Anthropic API call fails, retrieval falls back to the original query automatically.

The two entry points are:

- `QueryExpander.expand(query)` — synchronous
- `QueryExpander.expand_async(query)` — asynchronous

Both return a `list[str]` of expanded queries, or fall back to the original query on error.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `expand()` raises an exception | Whether `ANTHROPIC_API_KEY` is set and valid; pass `api_key=` explicitly to rule out environment issues |
| `expand()` returns a single-element list containing the original query | An API or JSON-parse error triggered the fallback — check network access to the Anthropic API and inspect the exception being swallowed |
| Expanded queries are unhelpful or off-topic | The model in use — confirm it is `claude-haiku-4-5-20251001` and that you have not overridden `model=` with a different string |
| Stale or repeated expansions across different queries | Response caching — `cache=True` by default; construct `QueryExpander(cache=False)` to disable it and confirm fresh results |
| `expand_async()` never resolves | Whether you are awaiting the coroutine in a running event loop; check for deadlocks if you are mixing sync and async code |
| Slow first call, fast subsequent calls | Expected behavior when `cache=True`; the first call hits the API and the result is cached |

## Step-by-step diagnosis

1. **Reproduce the failure with a minimal call.**
   Strip the invocation to its simplest form to confirm the failure is in `QueryExpander` itself and not in surrounding code:

   ```python
   from attune_rag.expander import QueryExpander

   expander = QueryExpander(api_key="YOUR_KEY", cache=False)
   print(expander.expand("how do I configure logging?"))
   ```

   If this works, the problem is in how your application constructs or calls `QueryExpander`.

2. **Check your API key and model name.**
   An invalid or missing key is the most common cause of silent fallback to the original query. Verify the key is accessible:

   ```bash
   echo $ANTHROPIC_API_KEY
   ```

   If the variable is empty, either export it or pass `api_key=` directly. Confirm the model string is exactly `claude-haiku-4-5-20251001`.

3. **Disable the cache.**
   If you are seeing stale results or cannot tell whether a real API call is being made, construct the expander with caching off:

   ```python
   expander = QueryExpander(cache=False)
   ```

   This forces a live API call on every `expand()` invocation.

4. **Enable DEBUG logging.**
   Set your logging level to `DEBUG` before instantiating `QueryExpander` to expose any exceptions being caught internally:

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

   Look for logged exceptions or unexpected JSON parse errors that indicate what the API actually returned.

5. **Inspect the raw API response.**
   The system prompt instructs the model to return only a JSON array of strings with no explanation and no markdown fences. If the model returns anything else, JSON parsing fails and the fallback triggers. Temporarily print or log the raw response to confirm the model is honoring the format.

6. **Run the expander test suite.**
   Confirm that the existing tests pass before investigating further:

   ```bash
   pytest -k "expander" -v
   ```

   A failing test that exercises your code path points directly at a regression.

## Common fixes

- **Missing API key.** Set the environment variable or pass it explicitly:

  ```bash
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```

  ```python
  expander = QueryExpander(api_key="sk-ant-...")
  ```

- **Silent fallback to original query.** This is almost always an API error or a JSON parse failure. Enable `DEBUG` logging (see step 4 above) to surface the underlying exception. Check that your network allows outbound HTTPS to Anthropic's API endpoints.

- **Unexpected model behavior.** If you overrode `model=` during construction, revert to the default:

  ```python
  expander = QueryExpander()  # uses claude-haiku-4-5-20251001
  ```

- **Cache returning wrong results.** Recreate the `QueryExpander` instance with `cache=False` to confirm whether the cache is serving a stale entry. If that resolves the issue, clear or bypass the cache for affected queries.

- **`anthropic` package not installed or wrong version.** `QueryExpander` depends on the `anthropic` Python SDK. Confirm it is installed:

  ```bash
  pip show anthropic
  ```

  If it is missing or outdated, install or upgrade it:

  ```bash
  pip install --upgrade anthropic
  ```

## Source files

- `src/attune_rag/expander.py`

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
