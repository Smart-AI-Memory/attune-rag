---
type: tip
name: reranker-tip
feature: reranker
depth: tip
generated_at: 2026-05-20T03:36:00.792516+00:00
source_hash: e8daea3d6507f630a80c7cd27c8ac195aab73bcc409c0b5cadc15e215ce9e11d
status: generated
---

# Tip: tune `candidate_multiplier` before changing anything else

Set `candidate_multiplier` to 3 (the default) or higher when you initialize `LLMReranker` — this controls how many keyword-retrieved candidates Claude Haiku evaluates before returning the final ranked list.

**Why:** The reranker can only surface a relevant document that keyword retrieval already found. A multiplier that is too low starves the LLM of candidates; raising it is the cheapest way to improve recall before touching queries or indexes.

**Tradeoff:** Each additional candidate adds latency within the 60-second timeout window. If you triple the multiplier on a large corpus, verify that your p95 response time stays within `timeout` (default `60.0` seconds), or pass a larger value explicitly:

```python
reranker = LLMReranker(
    candidate_multiplier=5,
    timeout=90.0,
)
```

The `rerank(query, hits)` method returns the same `RetrievalHit` objects in a new order — it does not filter them — so every index from the keyword stage is preserved in the output.

## Source files

- `src/attune_rag/reranker.py`

**Tags:** `reranker`, `hybrid-retrieval`, `precision`, `claude`, `haiku`
