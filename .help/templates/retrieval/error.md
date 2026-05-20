---
type: error
name: retrieval-error
feature: retrieval
depth: error
generated_at: 2026-05-20T03:22:04.682996+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Retrieval errors

## Common error signatures

Most retrieval failures fall into one of three categories: a corpus object that doesn't satisfy `CorpusProtocol`, a query that reduces to zero tokens after stopword filtering, or a `k` value that is incompatible with the scorer. The errors typically surface from `KeywordRetriever.retrieve()` or from a custom retriever that doesn't fully implement `RetrieverProtocol`.

Concrete signatures to watch for:

- **`AttributeError: 'XYZ' object has no attribute 'retrieve'`** — The object passed as a retriever doesn't implement `RetrieverProtocol`. Any retriever must expose `retrieve(query: str, corpus: CorpusProtocol, k: int = 3) -> Iterable[RetrievalHit]`.
- **`TypeError` on `retrieve()` call** — Argument types don't match the signature. Common causes: passing an integer where `query` expects a string, or passing a plain list where `corpus` expects a `CorpusProtocol` object.
- **Empty result list from `KeywordRetriever.retrieve()`** — Not an exception, but a silent failure. The query likely consists entirely of stopwords (for example, `"how do i"`, `"what is the"`), leaving no scored tokens to match against the corpus.

## Where errors originate

Check the class that matches your symptom before walking the call stack further.

- **`KeywordRetriever.retrieve(query, corpus, k)`** in `src/attune_rag/retrieval.py` — The most common raise site. This method tokenizes the query, strips stopwords from `_STOPWORDS`, applies suffix stemming via `_STEM_SUFFIXES`, and scores each `RetrievalEntry` using path, summary, content, and related-field weights. Failures here are usually caused by a malformed corpus or a query that produces no usable tokens.
- **`RetrieverProtocol`** in `src/attune_rag/retrieval.py` — A structural protocol, not a concrete class. If your retriever passes a runtime `isinstance` check but still raises, verify that its `retrieve()` return type is `Iterable[RetrievalHit]` and that each `RetrievalHit` carries a valid `RetrievalEntry`, a `float` score, and a non-empty `match_reason` string.
- **`RetrievalHit` construction** in `src/attune_rag/retrieval.py` — A dataclass with three required fields: `entry: RetrievalEntry`, `score: float`, and `match_reason: str`. Omitting any field or passing the wrong type raises a `TypeError` at construction time.

## How to diagnose

1. **Check whether the query survives stopword filtering.** `KeywordRetriever` removes every token found in `_STOPWORDS` (articles, modals, pronouns, and common prepositions such as `a`, `the`, `how`, `do`, `is`, `for`). If your entire query consists of stopwords, `retrieve()` returns an empty list rather than raising. Print the tokenized, filtered query before calling `retrieve()` to confirm at least one content token remains.

2. **Verify the corpus satisfies `CorpusProtocol`.** A corpus object that is missing expected attributes or iteration behavior causes `AttributeError` or `TypeError` inside `KeywordRetriever.retrieve()`. Confirm your corpus exposes the interface that `CorpusProtocol` requires before passing it to the retriever.

3. **Confirm `k` is a positive integer.** `KeywordRetriever.retrieve()` defaults to `k=3`. Passing `k=0` or a negative value may return an empty list or raise depending on how the scorer slices results. Pass an explicit, positive `k` to rule this out.

4. **Inspect `RetrievalHit.score` values when results are unexpectedly ranked.** `KeywordRetriever` weights token overlap across the `path`, `summary`, `content`, and `related` fields of each `RetrievalEntry`. A hit with a `score` of `0.0` means no stemmed query token matched any weighted field — check that the `RetrievalEntry` fields are populated and that stemming via `_STEM_SUFFIXES` (`-ing`, `-ed`, `-tion`, `-er`, and others) would produce a shared root with the query tokens.

5. **Trace a `TypeError` back to `RetrievalHit` construction.** If the traceback points inside `retrieval.py` at a dataclass instantiation, one of the three fields (`entry`, `score`, `match_reason`) is missing or the wrong type. Confirm the value passed as `score` is a `float` and `match_reason` is a non-empty string.

## Source files

- `src/attune_rag/retrieval.py`

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
