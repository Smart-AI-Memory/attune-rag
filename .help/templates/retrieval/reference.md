---
type: reference
name: retrieval-reference
feature: retrieval
depth: reference
generated_at: 2026-06-07T07:12:34.918465+00:00
source_hash: 7ee648ec170e0a70dd86867ff77d83f022fb08a9dcfd319a8f746f11101c98ed
status: generated
---

# Retrieval reference

Score and rank corpus entries against a query using `KeywordRetriever`, a token-overlap retriever that applies stemming, stopword filtering, and per-field weights across path, summary, content, and related fields.

## Classes

| Class | Description |
|-------|-------------|
| `RetrievalHit` | A single retrieval result holding the matched entry, its score, and the match reason. |
| `RetrieverProtocol` | Any object with a `retrieve(query, corpus, k)` method. |
| `KeywordRetriever` | Token-overlap retriever with path / summary / content / related weights. |

## `RetrievalHit`

`RetrievalHit` is a dataclass representing one item returned by a retriever.

### Fields

| Field | Type | Default |
|-------|------|---------|
| `entry` | `RetrievalEntry` | |
| `score` | `float` | |
| `match_reason` | `str` | |

## `RetrieverProtocol`

Structural protocol — any class that implements `retrieve(query, corpus, k)` satisfies it.

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `query: str`, `corpus: CorpusProtocol`, `k: int = 3` | `Iterable[RetrievalHit]` | Retrieve the top-`k` hits for `query` from `corpus`. |

## `KeywordRetriever`

Token-overlap retriever that scores entries by stemmed keyword matches, weighted by field (path, summary, content, related). Entries below `min_score` are excluded when a threshold is set.

### Constructor

| Parameters | Type | Default | Description |
|------------|------|---------|-------------|
| `min_score` | `float \| None` | `None` | Minimum score threshold; hits below this value are excluded from results. |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `query: str`, `corpus: CorpusProtocol`, `k: int = 3` | `list[RetrievalHit]` | Return the top-`k` scored and ranked `RetrievalHit` objects for `query`. |

## Constants

The following module-level constants control tokenization behavior during scoring.

### Stopwords

| Constant | Type | Members |
|----------|------|---------|
| `_STOPWORDS` | `frozenset` | `'a'`, `'an'`, `'the'`, `'how'`, `'do'`, `'does'`, `'i'`, `'to'`, `'with'`, `'for'`, `'is'`, `'are'`, `'of'`, `'in'`, `'on'`, `'at'`, `'and'`, `'or'`, `'but'`, `'can'`, `'should'`, `'would'`, `'will'`, `'be'`, `'been'`, `'by'`, `'my'`, `'me'`, `'we'`, `'it'`, `'this'`, `'that'`, `'these'`, `'those'` |

### Stem suffixes

| Constant | Type | Members (in order) |
|----------|------|--------------------|
| `_STEM_SUFFIXES` | `tuple` | `'ations'`, `'ation'`, `'ators'`, `'ator'`, `'ates'`, `'ate'`, `'ings'`, `'ing'`, `'ions'`, `'ion'`, `'ities'`, `'ity'`, `'ies'`, `'ers'`, `'ed'`, `'er'`, `'es'`, `'s'` |

## Source files

- `src/attune_rag/retrieval.py`

## Tags

`retrieval`, `keyword`, `scoring`, `ranking`
