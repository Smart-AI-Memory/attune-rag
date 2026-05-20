---
type: troubleshooting
name: reranker-troubleshooting
feature: reranker
depth: troubleshooting
generated_at: 2026-05-20T03:36:00.785508+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Troubleshoot reranker

## Before you start

`LLMReranker` sends your top-K keyword-retrieved candidates to Claude Haiku, which acts as a relevance judge and returns a re-ranked list as a JSON array of 0-based indices. If the Claude API call fails for any reason, `LLMReranker.rerank()` falls back silently to the original keyword-only order — no exception is raised to the caller.

Keep this fail-safe behavior in mind: if re-ranking appears to have no effect, the most likely cause is a suppressed API error, not a logic bug in your retrieval pipeline.

## Symptom table

| If you observe | Check |
|----------------|-------|
| Results are returned but order is identical to keyword retrieval | The Claude API call is failing silently — inspect the API response or enable error logging around the `rerank()` call |
| `rerank()` raises an exception | Your `hits` argument — confirm it is a `list[RetrievalHit]` and is not empty |
| `rerank()` times out | The `timeout` parameter (default `60.0` s) — increase it or check network connectivity to the Anthropic API |
| Returned list is missing candidates or contains duplicates | The Claude response — the model is required to return every 0-based index exactly once; a malformed JSON array causes index mapping to break |
| Re-ranking degrades rather than improves precision | The `candidate_multiplier` value — a multiplier that is too large sends low-quality candidates to the model; try reducing it from the default of `3` |
| `AuthenticationError` or `401` from Anthropic | The `api_key` argument or the `ANTHROPIC_API_KEY` environment variable — one must be set and valid |

## Step-by-step diagnosis

1. **Confirm the API key is available.**
   Check the environment before anything else — it is the most common cause of silent fallback:

   ```bash
   echo $ANTHROPIC_API_KEY
   ```

   If the variable is empty, either export it or pass `api_key=` explicitly when constructing `LLMReranker`.

2. **Reproduce the failure with a minimal call.**
   Strip the call down to its required arguments to confirm the failure occurs outside your application context:

   ```python
   from attune_rag.reranker import LLMReranker
   from attune_rag.retrieval import RetrievalHit  # adjust import as needed

   reranker = LLMReranker()  # uses ANTHROPIC_API_KEY from env
   hits = [RetrievalHit(...), RetrievalHit(...)]  # two real candidates
   result = reranker.rerank(query="version bump", hits=hits)
   print(result)
   ```

   If this call succeeds but your application fails, the problem is in how `hits` or `query` are constructed upstream.

3. **Check what Claude actually receives and returns.**
   Add temporary logging around `rerank()` to capture the raw API response before index mapping occurs. Confirm that:
   - The returned value is a valid JSON array.
   - Every index from `0` to `len(hits) - 1` appears exactly once.

   A response like `[0, 2]` when three candidates were sent is malformed and will produce incorrect results.

4. **Verify the model name and timeout.**
   `LLMReranker` defaults to `model='claude-haiku-4-5-20251001'` and `timeout=60.0`. If you have overridden either, confirm the values are valid:

   ```python
   reranker = LLMReranker(model='claude-haiku-4-5-20251001', timeout=60.0)
   ```

   An invalid model name causes an API error that triggers silent fallback to keyword order.

5. **Run the related tests.**
   Check whether existing tests still pass before modifying any code:

   ```bash
   pytest -k "reranker" -v
   ```

   A failing test that exercises the same path gives you a reproducible case and ready-made fixtures.

## Common fixes

- **Missing or invalid API key.** Set the key in your environment or pass it directly:

  ```bash
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```

  ```python
  reranker = LLMReranker(api_key="sk-ant-...")
  ```

- **Silent fallback masking an API error.** Because `rerank()` falls back to keyword order on any API failure, add explicit error handling in your calling code to surface the underlying exception during development:

  ```python
  import anthropic
  try:
      result = reranker.rerank(query=query, hits=hits)
  except anthropic.APIError as e:
      print(f"Reranker API error: {e}")
  ```

- **Timeout on slow networks.** Increase the timeout at construction time — this change is local to your `LLMReranker` instance:

  ```python
  reranker = LLMReranker(timeout=120.0)
  ```

- **Poor re-ranking quality from too many weak candidates.** Reduce `candidate_multiplier` so the model receives fewer but higher-quality candidates:

  ```python
  reranker = LLMReranker(candidate_multiplier=2)
  ```

- **Dependency version drift.** If re-ranking broke after a dependency update, confirm the installed Anthropic SDK version:

  ```bash
  pip show anthropic
  ```

  Roll back or pin the version if a recent upgrade changed API behavior.

## Source files

- `src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
