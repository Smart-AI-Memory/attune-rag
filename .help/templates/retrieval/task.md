---
type: task
name: retrieval-task
feature: retrieval
depth: task
generated_at: 2026-05-15T20:01:46.531123+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Work with retrieval

Use the retrieval module when you need to score and rank corpus entries against a query using token-overlap, stemming, and stopword filtering.

## Prerequisites

- Access to the project source code
- Familiarity with `src/attune_rag/retrieval.py`

## Understand the retrieval model

The module is built around three components in `src/attune_rag/retrieval.py`:

- **`RetrievalHit`** — A dataclass representing a single result. It holds a `RetrievalEntry`, a `score` float, and a `match_reason` string.
- **`RetrieverProtocol`** — A structural protocol. Any class that implements `retrieve(query, corpus, k)` and returns an `Iterable[RetrievalHit]` satisfies it.
- **`KeywordRetriever`** — The built-in implementation of `RetrieverProtocol`. It scores entries using token overlap across four weighted fields: path, summary, content, and related.

## Extend or implement retrieval

### Use `KeywordRetriever` directly

1. Import `KeywordRetriever` and your corpus from `src/attune_rag/retrieval.py`.
2. Instantiate `KeywordRetriever`.
3. Call `retrieve(query, corpus, k)`, where `query` is a string, `corpus` implements `CorpusProtocol`, and `k` is the number of results to return (default: `3`).
4. Iterate over the returned `list[RetrievalHit]` and read each hit's `entry`, `score`, and `match_reason`.

### Implement a custom retriever

1. Create a new class in your module.
2. Implement the method signature `retrieve(self, query: str, corpus: CorpusProtocol, k: int = 3) -> Iterable[RetrievalHit]`. This satisfies `RetrieverProtocol` without subclassing it.
3. Return `RetrievalHit` instances with a populated `entry`, a `score` between `0.0` and `1.0`, and a human-readable `match_reason`.

### Extend `KeywordRetriever`

1. Subclass `KeywordRetriever` in a new file.
2. Override `retrieve` to add pre- or post-processing around the parent implementation — for example, reranking hits or filtering by a score threshold.
3. Call `super().retrieve(query, corpus, k)` to preserve the base token-overlap scoring.

## Run the tests

After making any changes, run:

```bash
pytest -k "retrieval"
```

## Verify your work

You know retrieval is working correctly when:

- `retrieve` returns exactly `k` `RetrievalHit` objects (or fewer if the corpus has fewer than `k` entries).
- Each hit's `score` is a non-negative float and its `match_reason` is a non-empty string.
- `pytest -k "retrieval"` passes with no failures.
