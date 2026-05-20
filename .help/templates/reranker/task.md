---
type: task
name: reranker-task
feature: reranker
depth: task
generated_at: 2026-05-20T03:36:00.768713+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Use the LLM reranker

Use `LLMReranker` when you want to improve retrieval precision by re-scoring keyword search candidates with Claude Haiku before returning results to the user.

## Prerequisites

- A Claude API key (set via the `api_key` parameter or your environment)
- Access to `src/attune_rag/reranker.py`

## Instantiate and call the reranker

1. **Import `LLMReranker`** from `src/attune_rag/reranker.py`.

2. **Create an instance**, passing your API key and any tuning parameters:

   ```python
   reranker = LLMReranker(
       model="claude-haiku-4-5-20251001",  # default
       api_key="YOUR_API_KEY",
       candidate_multiplier=3,             # fetch 3× more candidates than needed
       timeout=60.0,                       # seconds before falling back to keyword order
   )
   ```

   - `candidate_multiplier` controls how many keyword-retrieved candidates are passed to Claude for scoring. A value of `3` means Claude sees three times as many hits as your final result count.
   - If the Claude API call fails or times out, `LLMReranker` automatically falls back to the original keyword-retrieval order — no exception is raised.

3. **Call `rerank`** with the user query and the list of `RetrievalHit` objects returned by your keyword retriever:

   ```python
   reranked_hits = reranker.rerank(query="version bump", hits=keyword_hits)
   ```

   Claude acts as a relevance judge, applying ranking rules such as preferring `tool-release-prep.md` for release-related queries and `tool-fix-test.md` for test-failure queries.

4. **Use the returned list** as your final result set. The hits are ordered from most to least relevant according to Claude's judgment.

## Run the tests

Verify your integration with:

```bash
pytest -k "reranker"
```

All tests should pass, and no test should make a live API call (mock the Claude client in your test fixtures).

## Confirm success

You know the reranker is working correctly when:

- `reranker.rerank(...)` returns the same `RetrievalHit` objects as the input, but in a different order that places semantically relevant results first.
- Passing an invalid API key (or simulating a timeout) returns the original keyword order unchanged, with no exception raised.

## Key files

- `src/attune_rag/reranker.py` — contains `LLMReranker` and the relevance-judge system prompt
