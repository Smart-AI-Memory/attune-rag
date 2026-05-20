---
type: tip
name: expander-tip
feature: expander
depth: tip
generated_at: 2026-05-20T03:34:53.279526+00:00
source_hash: 8d3d8b3b23e7c8dfce64df28069feb915c0f099b70568ec577cb0a58dddf9b78
status: generated
---

# Tip: Expand queries before retrieval, not after

Enable `QueryExpander` early in your retrieval pipeline, before you hit the vector store or keyword index. Claude Haiku rewrites your query into 3–5 alternative phrasings — surfacing feature names, tool categories, and developer jargon — so documents with low surface-level overlap with the original query still get a chance to rank.

**Why it's worth it:** A single terse query like "how do I cache responses" may never match docs that use the words "memoize", "persist", or "store results" — expansion closes that gap without changing your retrieval logic downstream.

**The tradeoff:** Each call to `expand()` or `expand_async()` makes a round-trip to the Claude Haiku API, adding latency and token cost. Pass `cache=True` (the default) to avoid re-expanding identical queries, but expect the first call on a cold cache to be slower. If the API call fails, `QueryExpander` falls back to the original query, so retrieval still works — you just lose the recall benefit.

## Source files

- `src/attune_rag/expander.py`

**Tags:** `expander`, `hybrid-retrieval`, `recall`, `claude`, `haiku`
