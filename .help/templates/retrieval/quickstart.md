---
type: quickstart
name: retrieval-quickstart
feature: retrieval
depth: quickstart
generated_at: 2026-05-20T03:22:04.695806+00:00
source_hash: 808240403d72c9dd7f4942d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Quickstart: retrieval

`KeywordRetriever` scores and ranks entries in a corpus against a query using token-overlap, stemming, and stopword filtering. The snippet below runs a retrieval and prints the top result.

```python
from attune_rag.retrieval import KeywordRetriever

retriever = KeywordRetriever()
hits = retriever.retrieve(query="configure logging", corpus=my_corpus, k=3)

for hit in hits:
    print(hit.score, hit.match_reason, hit.entry)
```

Expected output (values depend on your corpus):

```
0.87  keyword overlap: log, configur  <RetrievalEntry path='docs/logging.md'>
0.61  keyword overlap: configur       <RetrievalEntry path='docs/setup.md'>
0.44  keyword overlap: log            <RetrievalEntry path='docs/debug.md'>
```

## Prerequisites

- The project is cloned and installed locally.
- You have a corpus object that implements `CorpusProtocol`.

## Steps

1. **Create a retriever.** Instantiate `KeywordRetriever` — no required arguments.

   ```python
   from attune_rag.retrieval import KeywordRetriever
   retriever = KeywordRetriever()
   ```

2. **Run a query.** Call `retrieve(query, corpus, k)`. Set `k` to the number of results you want (default `3`). Common stopwords such as `"the"`, `"how"`, and `"is"` are filtered automatically; the retriever also stems tokens before scoring.

   ```python
   hits = retriever.retrieve(query="configure logging", corpus=my_corpus, k=3)
   ```

3. **Inspect the hits.** Each returned `RetrievalHit` exposes three fields: `entry` (the matched corpus entry), `score` (float), and `match_reason` (a human-readable explanation of why the entry ranked).

   ```python
   for hit in hits:
       print(hit.score, hit.match_reason, hit.entry)
   ```

## Source files

- `src/attune_rag/retrieval.py`

---

Next: Swap `KeywordRetriever` for your own retriever by implementing the `RetrieverProtocol` — any class with a `retrieve(query, corpus, k)` method that returns an iterable of `RetrievalHit` objects will work as a drop-in replacement.

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
