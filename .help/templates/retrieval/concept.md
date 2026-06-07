---
type: concept
name: retrieval-concept
feature: retrieval
depth: concept
generated_at: 2026-06-07T07:12:34.906058+00:00
source_hash: 7ee648ec170e0a70dd86867ff77d83f022fb08a9dcfd319a8f746f11101c98ed
status: generated
---

# Retrieval

Retrieval is the process of scoring and ranking corpus entries against a query so that the most relevant results surface first.

## How keyword retrieval works

`KeywordRetriever` implements retrieval through token overlap: it compares the meaningful words in a query against the words in each corpus entry, then returns the top `k` matches as a ranked list of `RetrievalHit` objects.

Before comparing tokens, the retriever strips common stopwords — words like `"the"`, `"is"`, and `"for"` that carry no signal — and reduces remaining words to their stems by removing suffixes such as `"ing"`, `"ation"`, and `"er"`. This means a query for `"configuring"` matches entries that contain `"configuration"` or `"configure"`.

Scoring is not uniform across an entry. The retriever weights matches differently depending on where in the entry the token appears — for example, a match in a path or summary contributes differently than a match in body content or a related-terms field. The final `score` on each `RetrievalHit` reflects this weighted overlap.

You can set a `min_score` threshold when constructing a `KeywordRetriever` to discard low-confidence results before they reach the caller:

```python
retriever = KeywordRetriever(min_score=0.25)
hits = retriever.retrieve("template design", corpus, k=5)
```

## The three pieces and how they fit together

| Piece | Role |
|---|---|
| `RetrieverProtocol` | The interface any retriever must satisfy: a `retrieve(query, corpus, k)` method returning `Iterable[RetrievalHit]`. Code that consumes results depends on this protocol, not on `KeywordRetriever` directly. |
| `KeywordRetriever` | The concrete implementation. Accepts an optional `min_score` floor and returns a `list[RetrievalHit]` sorted by score. |
| `RetrievalHit` | A single ranked result. Carries the matched `entry`, its `score`, and a `match_reason` string that explains why the entry was selected. |

Because callers type against `RetrieverProtocol`, you can substitute any retriever — semantic, hybrid, or otherwise — without changing the code that processes `RetrievalHit` objects.

## When retrieval matters

Retrieval sits between a raw user query and a final answer. It determines which corpus entries a downstream component sees, so the quality of retrieval directly bounds the quality of any response built on top of it. If `k` is too small or `min_score` is too high, relevant entries are excluded. If `min_score` is absent or very low, noisy entries pass through. Tuning these two parameters is the primary lever for controlling precision and recall in your pipeline.
