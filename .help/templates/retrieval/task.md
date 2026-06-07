---
type: task
name: retrieval-task
feature: retrieval
depth: task
generated_at: 2026-06-07T07:12:34.913261+00:00
source_hash: 7ee648ec170e0a70dd86867ff77d83f022fb08a9dcfd319a8f746f11101c98ed
status: generated
---

# Work with retrieval

Use `KeywordRetriever` when you need to search a corpus by keyword and get back ranked results — for example, to surface the most relevant entries before passing them to a language model.

## Prerequisites

- `attune_rag` installed and importable in your environment
- A corpus object that satisfies `CorpusProtocol`

## Steps

1. **Import the retrieval classes.**

   ```python
   from attune_rag.retrieval import KeywordRetriever, RetrievalHit
   ```

2. **Instantiate `KeywordRetriever`.**

   Pass `min_score` if you want to discard low-confidence results. Omit it to return all scored hits.

   ```python
   retriever = KeywordRetriever(min_score=0.1)
   ```

3. **Call `retrieve` with your query, corpus, and result count.**

   `k` controls how many hits are returned (default: `3`).

   ```python
   hits: list[RetrievalHit] = retriever.retrieve(
       query="how to configure logging",
       corpus=my_corpus,
       k=5,
   )
   ```

4. **Inspect each `RetrievalHit`.**

   Each hit exposes three fields:

   | Field | Type | Description |
   |---|---|---|
   | `entry` | `RetrievalEntry` | The matched corpus entry |
   | `score` | `float` | Relevance score assigned by the retriever |
   | `match_reason` | `str` | Human-readable explanation of why this entry matched |

   ```python
   for hit in hits:
       print(hit.score, hit.match_reason, hit.entry)
   ```

5. **Swap in a custom retriever if needed.**

   If `KeywordRetriever` does not suit your use case, implement `RetrieverProtocol` instead. Any class with a `retrieve(self, query: str, corpus: CorpusProtocol, k: int = 3) -> Iterable[RetrievalHit]` method satisfies the protocol and can be used anywhere `KeywordRetriever` is accepted.

   ```python
   class MyRetriever:
       def retrieve(self, query: str, corpus: CorpusProtocol, k: int = 3):
           ...  # your logic here
   ```

## Verify the result

After calling `retrieve`, confirm that:

- `hits` is a non-empty list when the corpus contains relevant entries.
- Each `hit.score` is greater than or equal to the `min_score` you set (if any).
- Each `hit.match_reason` is a non-empty string describing the match.

If `hits` is empty and you expect matches, lower `min_score` or omit it entirely to check whether the retriever is scoring entries at all.
