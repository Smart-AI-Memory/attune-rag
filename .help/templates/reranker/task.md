---
type: task
name: reranker-task
feature: reranker
depth: task
generated_at: 2026-05-20T02:45:30.125433+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Work with the reranker

Use the reranker when you want to improve retrieval precision by re-ranking keyword search candidates with Claude Haiku before returning results to the user.

## Prerequisites

- Read access to `src/attune_rag/reranker.py`
- A Claude API key, or confirm that the calling code passes one explicitly
- `pytest` installed and runnable in your environment

## Steps

1. **Open the reranker module.**
   Open `src/attune_rag/reranker.py` and locate `LLMReranker`. Note its constructor signature:

   ```python
   LLMReranker(
       model: str = 'claude-haiku-4-5-20251001',
       api_key: str | None = None,
       candidate_multiplier: int = 3,
       timeout: float = 60.0,
   )
   ```

   `candidate_multiplier` controls how many keyword candidates are fetched before re-ranking. `timeout` caps the Claude API call; any error causes `rerank()` to return the original keyword order unchanged.

2. **Instantiate `LLMReranker` with your settings.**
   Pass an explicit `api_key` if your environment does not expose it through the default lookup. Adjust `candidate_multiplier` if you want to widen or narrow the candidate pool fed to Claude:

   ```python
   reranker = LLMReranker(
       api_key="YOUR_API_KEY",
       candidate_multiplier=5,
   )
   ```

3. **Call `rerank()` with a query and retrieval hits.**
   Pass the user query string and the list of `RetrievalHit` objects returned by your keyword search. `rerank()` returns a new list sorted from most to least relevant:

   ```python
   ranked_hits = reranker.rerank(query="version bump", hits=keyword_hits)
   ```

   If the Claude API call fails or times out, `rerank()` returns `hits` in its original order — no exception is raised.

4. **Extend `LLMReranker` if you need custom ranking logic.**
   Create a subclass in `src/attune_rag/reranker.py` rather than editing `LLMReranker` directly. Override `rerank()` and keep the same method signature so callers stay compatible.

5. **Run the reranker tests.**
   Confirm your changes pass the existing suite:

   ```bash
   pytest -k "reranker"
   ```

## Verify success

All targeted tests pass with no failures or errors. When you call `rerank()` with a query such as `"version bump"`, the returned list places hits whose paths contain `tool-release-prep` ahead of hits matching `task-ci-cd-pipeline` — confirming that Claude's relevance judgement is taking effect rather than the fallback keyword order.

## Key files

- `src/attune_rag/reranker.py` — contains `LLMReranker` and the `_SYSTEM` prompt that instructs Claude how to rank candidates
