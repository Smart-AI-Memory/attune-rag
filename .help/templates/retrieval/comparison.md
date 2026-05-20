---
type: comparison
name: retrieval-comparison
feature: retrieval
depth: comparison
generated_at: 2026-05-20T03:22:04.702288+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Comparison: KeywordRetriever vs custom retriever implementations

## Context

The retrieval module gives you two things: a ready-to-use `KeywordRetriever` that scores corpus entries against a query using token overlap, stemming, and stopword filtering, and a `RetrieverProtocol` that lets you swap in any retriever that satisfies the same `retrieve(query, corpus, k)` interface.

Choosing between them comes down to one question: does `KeywordRetriever`'s scoring model fit your data?

## Feature comparison

| Capability | `KeywordRetriever` | Custom `RetrieverProtocol` impl |
|---|---|---|
| Setup effort | Zero — instantiate and call | Must implement `retrieve(query, corpus, k)` |
| Scoring model | Token-overlap weighted across `path`, `summary`, `content`, and `related` fields | Whatever you define |
| Stopword filtering | Built-in 35-word list (`a`, `the`, `how`, `does`, …) | Your responsibility |
| Stemming | Strips 16 suffix patterns (`-ation`, `-ing`, `-ed`, `-er`, `-es`, …) | Your responsibility |
| Return type | `list[RetrievalHit]` (ordered, indexable) | `Iterable[RetrievalHit]` (any iterable) |
| Field-weight tuning | Fixed weights per field; not configurable at runtime | Fully configurable |
| Semantic / embedding search | Not supported | Supported — implement it yourself |
| Drop-in replaceability | Yes — satisfies `RetrieverProtocol` | Yes — anything with the right signature qualifies |

## Scoring model details (KeywordRetriever)

`KeywordRetriever` tokenizes both the query and each `RetrievalEntry`, removes stopwords, applies suffix-stripping, then accumulates overlap scores weighted by field:

- **`path`** — where the entry lives in the corpus
- **`summary`** — short description of the entry
- **`content`** — full body text
- **`related`** — linked or associated entries

Each `RetrievalHit` records the final `score` (float) and a `match_reason` string explaining which tokens drove the match. The top-`k` hits are returned (default `k=3`).

Because the model is purely lexical, it degrades when query terms and entry text use different but synonymous vocabulary (for example, "configure" vs. "set up").

## When NOT to use KeywordRetriever

- **Your queries are semantic, not lexical.** `KeywordRetriever` has no embedding or vector support. If users phrase queries differently from the corpus text, token overlap produces poor rankings.
- **You need runtime weight tuning.** The per-field weights are fixed in the implementation. If you need to boost `summary` over `content` based on context, you need a custom implementation.
- **You are integrating a third-party retrieval backend.** Wrap it behind `RetrieverProtocol` rather than forcing it through `KeywordRetriever`'s scoring logic.
- **You are doing a one-off exploratory script.** Wiring up a full `CorpusProtocol` for a throwaway use case is likely more overhead than the task warrants.

## Use X when…

**Use `KeywordRetriever` when:**
- Your corpus entries have reliable `path`, `summary`, `content`, or `related` text and your queries share vocabulary with that text.
- You want retrieval working immediately with no scoring code to write or maintain.
- Lexical precision matters more than recall across paraphrased queries.

**Implement a custom `RetrieverProtocol` when:**
- You need semantic, embedding-based, or hybrid retrieval.
- You need to adjust field weights dynamically or score fields that `KeywordRetriever` does not cover.
- You are wrapping an external search service (for example, a vector database or full-text search engine) and want it to be interchangeable with other retrievers in the same pipeline.

For most corpus-search tasks where the query vocabulary matches the corpus text, `KeywordRetriever` is the right starting point — it handles stopword filtering and stemming for you and satisfies `RetrieverProtocol`, so you can replace it later without changing call sites.

## Source files

- `src/attune_rag/retrieval.py`

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
