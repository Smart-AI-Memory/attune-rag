---
type: concept
name: retrieval-concept
feature: retrieval
depth: concept
generated_at: 2026-05-15T20:01:46.526925+00:00
source_hash: 808240403d72c9dd7f4956d5cbde040fffbec4b4befa504fb60f9fd862
status: generated
---

# Retrieval

Retrieval is the process of scoring and ranking help templates against a user query so that the most relevant entries surface first.

## How retrieval works

When you submit a query, `KeywordRetriever` compares your query tokens against every `RetrievalEntry` in the corpus and returns the top `k` matches as a ranked list of `RetrievalHit` objects.

The scoring pipeline has three stages:

1. **Tokenization and filtering.** The query string is split into tokens. Common words — articles, prepositions, modal verbs, and pronouns such as `a`, `the`, `how`, `should`, and `we` — are removed using the `_STOPWORDS` set, so only content-bearing terms remain.

2. **Stemming.** Each remaining token is reduced to its root form by stripping known suffixes (`_STEM_SUFFIXES`), so `migrations`, `migrating`, and `migrated` all match on `migrat`. This lets a query for "configuring" find entries that use "configuration" or "configured."

3. **Weighted overlap scoring.** The retriever compares stemmed query tokens against four fields of each entry — `path`, `summary`, `content`, and `related` — and weights them differently. A match in a template's `path` or `summary` scores higher than a match buried in body `content`, reflecting the assumption that titles and summaries are more authoritative signals of relevance.

The result is a list of `RetrievalHit` objects sorted by descending `score`. Each hit carries the matched `RetrievalEntry`, its numeric score, and a `match_reason` string explaining why it ranked where it did.

## Core components

| Component | Role |
|---|---|
| `KeywordRetriever` | Implements the token-overlap algorithm. Accepts a `query` string, a `corpus` (any `CorpusProtocol`), and an integer `k`, and returns the top-k `RetrievalHit` objects. |
| `RetrieverProtocol` | The structural interface any retriever must satisfy: a `retrieve(query, corpus, k)` method returning an iterable of `RetrievalHit`. Swap in a different retriever by implementing this protocol. |
| `RetrievalHit` | A dataclass wrapping one result: the `RetrievalEntry` that matched, its `score`, and the `match_reason`. |

## When retrieval matters

Retrieval runs every time the engine needs to answer "which templates are relevant to this query?" It sits between the corpus (the full set of stored templates) and whatever consumes ranked results — for example, the component that decides which templates to render or surface to the user.

If results feel off — relevant templates ranking low, or unrelated templates appearing at the top — the weighted scoring fields and the stopword and suffix lists are the primary levers to examine.
