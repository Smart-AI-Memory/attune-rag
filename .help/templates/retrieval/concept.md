---
type: concept
feature: retrieval
depth: concept
generated_at: 2026-04-23T03:33:16.988868+00:00
source_hash: 7143f387f3dccfded707adcfa52af1fdc50a71361e9de5a4bd466bc191c3f35b
status: generated
---

# Retrieval

Retrieval finds the most relevant help entries for a user's query by scoring text overlap between the query and corpus entries.

## How keyword retrieval works

The system uses token-based matching to score relevance. When you submit a query, the `KeywordRetriever` breaks it into meaningful words, removes common stopwords like "the" and "how," and applies light stemming to match variations like "migrate" and "migration."

Each corpus entry gets scored based on four weighted components:

- **Path matching** — file names and directory structure
- **Summary matching** — brief descriptions and titles
- **Content matching** — full text content
- **Related terms matching** — associated tags and cross-references

The retriever ranks all matches by combined score and returns the top k results as `RetrievalHit` objects containing the entry, score, and explanation of why it matched.

## Retrieval components

**`RetrievalHit`** represents a single search result with the matched entry, numerical score, and human-readable match reason.

**`RetrieverProtocol`** defines the interface that all retrievers must implement: a `retrieve()` method that takes a query string, corpus, and result count.

**`KeywordRetriever`** implements token-overlap scoring with configurable weights for different text components. It handles stopword filtering using a predefined set of 35 common words and applies suffix-based stemming for 16 common endings.

## Text processing details

The retriever normalizes text by removing stopwords like "a," "an," "the," "how," "do," and "i" before scoring. It also applies light stemming by stripping suffixes like "-ation," "-ing," "-ed," and "-er" to match word variations.

This preprocessing helps queries like "migrating templates" match content about "template migration" without requiring exact phrase matches.
