---
type: quickstart
name: reranker-quickstart
feature: reranker
depth: quickstart
generated_at: 2026-05-20T03:36:00.790210+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Quickstart: reranker

`LLMReranker` uses Claude Haiku as a relevance judge over your keyword-retrieved candidates, returning them in precision-optimized order. If the Claude API call fails for any reason, results fall back to the original keyword order automatically.

```python
from attune_rag.reranker import LLMReranker

reranker = LLMReranker(api_key="YOUR_ANTHROPIC_API_KEY")
reranked = reranker.rerank(query="version bump and publish to PyPI", hits=my_hits)
print([hit.path for hit in reranked])
```

Expected output — your hits reordered from most to least relevant:

```
['concepts/tool-release-prep.md', 'task-package-publishing.md', ...]
```

## Prerequisites

- `attune-rag` installed in your local environment
- An Anthropic API key with access to Claude Haiku (`claude-haiku-4-5-20251001`)
- A list of `RetrievalHit` objects from a prior keyword retrieval step

## Steps

1. **Instantiate `LLMReranker`.** Pass your API key directly or set it via the environment. The `candidate_multiplier` parameter (default `3`) controls how many keyword candidates are retrieved per result you ultimately want.

   ```python
   reranker = LLMReranker(
       api_key="YOUR_ANTHROPIC_API_KEY",
       candidate_multiplier=3,
       timeout=60.0,
   )
   ```

2. **Call `rerank()` with your query and hits.** Pass the user's original query string and the list of `RetrievalHit` objects returned by your keyword retrieval step.

   ```python
   reranked_hits = reranker.rerank(
       query="how do I fix a failing CI pipeline?",
       hits=keyword_hits,
   )
   ```

3. **Use the reordered results.** `rerank()` returns the same `RetrievalHit` list sorted from most to least relevant. The first item is Claude's top pick for the query.

   ```python
   for hit in reranked_hits:
       print(hit.path)
   ```

   Expected output for a "fix failing CI pipeline" query:

   ```
   concepts/tool-fix-test.md
   skill-fix-test.md
   task-ci-cd-pipeline.md
   ```

## Source files

- `src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`

---

Next: Read the `_SYSTEM` prompt in `src/attune_rag/reranker.py` to understand the ranking heuristics Claude applies — knowing which path prefixes (`tool-`, `skill-`, `task-`, `use-`) it favors for different query types helps you structure your document paths for maximum retrieval precision.
