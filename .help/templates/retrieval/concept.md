---
type: concept
name: retrieval-concept
feature: retrieval
depth: concept
generated_at: 2026-05-20T03:22:04.667571+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Retrieval

## Overview

Retrieval is the process of scoring and ranking corpus entries against a query, returning the top-k matches as `RetrievalHit` objects. The default implementation uses token overlap — stripping stopwords and stemming query and entry tokens — to produce a ranked list of results.

## Mental model

A retriever takes three inputs — a query string, a corpus, and a result count `k` — and returns an ordered list of `RetrievalHit` objects. Each hit wraps a `RetrievalEntry` together with a numeric `score` and a human-readable `match_reason` that explains why the entry ranked.

`KeywordRetriever` applies this pattern using weighted token overlap. It tokenizes the query, removes common stopwords (such as `a`, `the`, `how`, and `is`), and strips suffixes (such as `ation`, `ing`, and `er`) to normalize tokens before comparing them against each entry's path, summary, content, and related fields. Each field carries its own weight, so a match in a path or summary can outrank the same token found only in body content.

```
query ──► tokenize & stem ──► compare against corpus entries
                                   │
                         path · summary · content · related
                         (weighted token overlap per field)
                                   │
                              score + match_reason
                                   │
                           top-k RetrievalHit list
```

## Core types

| Type | Role |
|------|------|
| `RetrievalHit` | A single ranked result, holding a `RetrievalEntry`, a `float` score, and a `match_reason` string. |
| `RetrieverProtocol` | The structural interface any retriever must satisfy: a `retrieve(query, corpus, k)` method that returns an iterable of `RetrievalHit` objects. |
| `KeywordRetriever` | The built-in implementation of `RetrieverProtocol`, using stopword filtering and suffix stemming to compute token-overlap scores. |

## Token normalization

`KeywordRetriever` normalizes tokens in two steps before scoring:

1. **Stopword removal** — tokens in `_STOPWORDS` (for example, `and`, `or`, `can`, `should`) are discarded so they do not inflate match scores.
2. **Suffix stemming** — suffixes in `_STEM_SUFFIXES` are stripped in length order (for example, `ations` → `ation` → `ate`) so that `configure`, `configuring`, and `configuration` resolve to the same root token.

## Extension point

Because `RetrieverProtocol` is a structural protocol, any class that implements `retrieve(query, corpus, k)` with the correct signature works as a drop-in replacement for `KeywordRetriever`. You do not need to inherit from a base class; satisfying the method signature is sufficient.
