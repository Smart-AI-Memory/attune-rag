---
type: task
name: reranker-task
feature: reranker
depth: task
generated_at: 2026-07-10T13:06:04.026407+00:00
source_hash: c828b6c3ccd4f66d997d42c41fc386540dc0978c13f780db0e2320cdbb911f6d
status: generated
---

# Work with the reranker

Use the reranker when you want Claude Haiku to re-rank keyword retrieval candidates by relevance, improving retrieval precision beyond what keyword search alone provides.

## Prerequisites

- Access to `src/attune_rag/reranker.py`
- A valid Anthropic API key (or the `ANTHROPIC_API_KEY` environment variable set)

## Steps

1. **Locate `LLMReranker` in `src/attune_rag/reranker.py`.**
   This is the only class in the module. It accepts four constructor parameters:
   - `model` — Claude model string; defaults to Claude Haiku when `None`
   - `api_key` — Anthropic API key; falls back to the environment variable when `None`
   - `candidate_multiplier` — multiplies the number of keyword candidates passed to Claude before re-ranking (default: `3`)
   - `timeout` — seconds before the API call times out (default: `60.0`)

2. **Instantiate `LLMReranker` with your chosen settings.**
   Pass only the parameters you need to override:

   ```python
   from attune_rag.reranker import LLMReranker

   reranker = LLMReranker(candidate_multiplier=5, timeout=30.0)
   ```

3. **Call `rerank(query, hits)` with your query string and retrieved hits.**
   `hits` is a list of `RetrievalHit` objects returned by your keyword retrieval step:

   ```python
   reranked = reranker.rerank(query="how do I publish to PyPI?", hits=keyword_hits)
   ```

   The method returns the same `RetrievalHit` list ordered from most to least relevant. If the Claude API call fails for any reason, `rerank` returns the original `hits` list unchanged (keyword-order fallback).

4. **Extend `LLMReranker` if you need custom ranking logic.**
   Subclass `LLMReranker` and override `rerank` rather than modifying the base class directly. Match the existing naming, error-handling, and logging style used in `reranker.py`.

5. **Run the reranker tests to verify your changes.**

   ```bash
   pytest -k "reranker"
   ```

## Verify success

All `reranker` tests pass and `rerank` returns a reordered list. To confirm Claude is driving the order (rather than the fallback), check that the most semantically relevant hit appears first even when it ranked lower in the raw keyword results.
