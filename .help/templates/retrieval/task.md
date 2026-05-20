---
type: task
name: retrieval-task
feature: retrieval
depth: task
generated_at: 2026-05-20T03:22:04.674108+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Work with retrieval

Use keyword retrieval when you need to score and rank corpus entries against a natural-language query — `KeywordRetriever` filters stopwords, applies suffix stemming, and weights matches across an entry's path, summary, content, and related fields to return the top-k `RetrievalHit` results.

## Prerequisites

- Read access to `src/attune_rag/retrieval.py`
- A corpus object that satisfies `CorpusProtocol`

## Understand the retrieval model

Before writing code, familiarise yourself with the three building blocks in `src/attune_rag/retrieval.py`:

| Class | Role |
|---|---|
| `RetrievalHit` | Dataclass wrapping a single result: a `RetrievalEntry`, a `float` score, and a `str` match reason. |
| `RetrieverProtocol` | Structural protocol — any object that exposes `retrieve(query, corpus, k)` qualifies. |
| `KeywordRetriever` | Concrete token-overlap retriever. Strips `_STOPWORDS`, stems tokens using `_STEM_SUFFIXES`, then scores each entry by weighted field overlap. |

## Use `KeywordRetriever` directly

1. Instantiate `KeywordRetriever` — it requires no constructor arguments.

   ```python
   from attune_rag.retrieval import KeywordRetriever

   retriever = KeywordRetriever()
   ```

2. Call `retrieve` with your query string, a `CorpusProtocol` object, and the number of results you want.

   ```python
   hits = retriever.retrieve(query="authentication tokens", corpus=my_corpus, k=5)
   ```

3. Iterate over the returned `list[RetrievalHit]` and read each hit's fields.

   ```python
   for hit in hits:
       print(hit.score, hit.match_reason, hit.entry)
   ```

   **Success criterion:** `hits` is a list of up to `k` `RetrievalHit` objects ordered from highest to lowest score. If the list is empty, your query tokens were all stopwords or produced no overlap with the corpus.

## Implement a custom retriever

1. Create a class that implements the `RetrieverProtocol` signature — no explicit inheritance is required.

   ```python
   from collections.abc import Iterable
   from attune_rag.retrieval import RetrievalHit

   class SemanticRetriever:
       def retrieve(
           self,
           query: str,
           corpus: CorpusProtocol,
           k: int = 3,
       ) -> Iterable[RetrievalHit]:
           ...
   ```

2. Return `RetrievalHit` instances from your `retrieve` method. Populate `entry` with the matched `RetrievalEntry`, `score` with a numeric relevance value, and `match_reason` with a human-readable explanation.

3. Pass your retriever wherever a `RetrieverProtocol` is expected — duck typing means no cast or adapter is needed.

   **Success criterion:** Static type checkers and any runtime protocol checks accept your class without errors.

## Run the tests

Verify your changes against the existing test suite:

```shell
pytest -k "retrieval"
```

All tests pass when the output shows `no failed` for the `retrieval` suite.

## Key files

- `src/attune_rag/retrieval.py` — contains `RetrievalHit`, `RetrieverProtocol`, `KeywordRetriever`, `_STOPWORDS`, and `_STEM_SUFFIXES`
