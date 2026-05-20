---
type: faq
name: retrieval-faq
feature: retrieval
depth: faq
generated_at: 2026-05-20T03:22:04.693256+00:00
source_hash: 808240403d72c9dd7f4968c2816840fffbec4b4befa509d3a494fb60f9fd862
status: generated
---

# Retrieval FAQ

## What does the retrieval module do?

It ranks entries from a corpus against a text query using token overlap, stemming, and stopword filtering. The top results are returned as `RetrievalHit` objects, each carrying the matched entry, a score, and a reason for the match.

## When should I use `KeywordRetriever`?

Use it when you need lightweight, lexical search over a corpus without a vector store or embedding model. It's a good fit for small-to-medium corpora where keyword overlap is a reliable signal.

## What's the difference between `KeywordRetriever` and `RetrieverProtocol`?

`RetrieverProtocol` is a structural protocol — any class that implements `retrieve(query, corpus, k)` satisfies it. `KeywordRetriever` is a concrete implementation of that protocol that scores entries by weighted token overlap across path, summary, content, and related fields.

## What does a `RetrievalHit` contain?

Three fields:

- `entry` — the matched `RetrievalEntry` from the corpus
- `score` — a float representing the match strength
- `match_reason` — a string explaining why the entry was selected

## How does `KeywordRetriever` score entries?

It tokenizes the query, strips stopwords (such as "the", "how", "is"), and applies a simple suffix-stripping stemmer (removing endings like "-ing", "-ation", "-ed"). It then computes token overlap against each entry's path, summary, content, and related fields using per-field weights, and returns the top `k` results.

## Which words does the retriever ignore?

It filters out a fixed set of common stopwords including articles, prepositions, and modal verbs — for example: `a`, `an`, `the`, `how`, `is`, `can`, `should`, `for`, `with`. Short function words that rarely carry meaning are excluded so they don't inflate scores.

## How do I control how many results are returned?

Pass the `k` argument to `retrieve()`. It defaults to `3`:

```python
hits = retriever.retrieve(query, corpus, k=10)
```

## How do I debug unexpected results?

1. Run `pytest -k "retrieval" -v` to confirm the module itself is working correctly.
2. Inspect `hit.score` and `hit.match_reason` on the returned `RetrievalHit` objects to understand how entries are being ranked.
3. If query terms are being dropped unexpectedly, check whether they appear in `_STOPWORDS` or are being collapsed by the stemmer's suffix list (`_STEM_SUFFIXES`).
4. Add a `logger.debug` call before `retrieve()` to log the tokenized query and confirm what terms are actually being matched.

## Where is the source code?

All retrieval logic lives in `src/attune_rag/retrieval.py`.

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
