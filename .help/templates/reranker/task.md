---
type: task
name: reranker-task
feature: reranker
depth: task
generated_at: 2026-06-07T07:14:09.698838+00:00
source_hash: d9cc73a55820ef60156edf63a24310f219daaa440a814d281fee2195484a90ae
status: generated
---

# Rerank retrieval results with LLMReranker

Use `LLMReranker` when keyword retrieval returns too many loosely relevant hits and you need Claude to score and reorder them by relevance before presenting results to the user.

## Prerequisites

- An Anthropic API key with access to `claude-haiku-4-5`
- A list of `RetrievalHit` objects from your keyword retrieval step
- `attune_rag.reranker` available in your Python environment

## Steps

1. **Import `LLMReranker` from `attune_rag.reranker`.**

   ```python
   from attune_rag.reranker import LLMReranker
   ```

2. **Instantiate `LLMReranker` with your configuration.**

   Pass your API key and, optionally, adjust `candidate_multiplier` to control how many candidates Claude evaluates relative to the number of results you want returned. The default `candidate_multiplier` is `3` and the default `timeout` is `60.0` seconds.

   ```python
   reranker = LLMReranker(
       model="claude-haiku-4-5",
       api_key="YOUR_API_KEY",
       candidate_multiplier=3,
       timeout=60.0,
   )
   ```

3. **Call `rerank` with your query and retrieved hits.**

   Pass the user query string and the list of `RetrievalHit` objects from your retrieval step. `rerank` returns a new list of `RetrievalHit` objects sorted from most to least relevant.

   ```python
   reranked_hits = reranker.rerank(query=user_query, hits=keyword_hits)
   ```

   If the API call fails for any reason, `rerank` falls back to the original keyword-retrieval order, so your application continues to return results.

4. **Use the reranked list in your response pipeline.**

   Replace your existing hit list with `reranked_hits`. The first item in the list is the hit Claude judged most relevant to the query.

5. **Run the reranker tests to confirm nothing is broken.**

   ```shell
   pytest -k "reranker"
   ```

## Verify the task worked

After calling `rerank`, inspect the first element of the returned list and confirm it corresponds to the document you would expect to be most relevant for your test query. If Claude's judgment differs from keyword rank order, the reranker is working. If the API is unreachable, the returned list should match your original `hits` order exactly, confirming the fallback behavior is active.
