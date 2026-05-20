---
type: tip
name: retrieval-tip
feature: retrieval
depth: tip
generated_at: 2026-05-20T03:22:04.698108+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Tip: Plug in a custom retriever without subclassing `KeywordRetriever`

Implement `RetrieverProtocol` — a single `retrieve(query, corpus, k)` method — instead of extending `KeywordRetriever` when you need different scoring logic.

**Why:** `KeywordRetriever` bundles stopword filtering, suffix stemming, and four field weights (path, summary, content, related) into one concrete class. Subclassing it means inheriting all of that behavior and working around what you don't want. A fresh `RetrieverProtocol` implementation gives you a clean slate with the same interface.

**Tradeoff:** You lose `KeywordRetriever`'s token-overlap scoring for free. If you only need to adjust field weights, check whether `KeywordRetriever` already accepts weight configuration before writing a new retriever from scratch.

## Source files

- `src/attune_rag/retrieval.py`

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
