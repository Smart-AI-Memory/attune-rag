---
type: reference
name: retrieval-reference
feature: retrieval
depth: reference
generated_at: 2026-05-15T20:01:46.533886+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Retrieval reference

Score and rank `RetrievalEntry` objects against a query using `KeywordRetriever`, or implement `RetrieverProtocol` to supply a custom retriever.

## Classes

| Class | Description |
|-------|-------------|
| `RetrievalHit` | A single retrieval result. |
| `RetrieverProtocol` | Any object with a `retrieve(query, corpus, k)` method. |
| `KeywordRetriever` | Token-overlap retriever with path / summary / content / related weights. |

### `RetrievalHit`

`[dataclass]`

| Field | Type | Default |
|-------|------|---------|
| `entry` | `RetrievalEntry` | |
| `score` | `float` | |
| `match_reason` | `str` | |

### `RetrieverProtocol`

Any object that implements this method satisfies the protocol.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `self, query: str, corpus: CorpusProtocol, k: int = 3` | `Iterable[RetrievalHit]` | Retrieve the top-`k` hits from `corpus` for the given `query`. |

### `KeywordRetriever`

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `self, query: str, corpus: CorpusProtocol, k: int = 3` | `list[RetrievalHit]` | Retrieve the top-`k` hits from `corpus` using token-overlap scoring across path, summary, content, and related fields. |

## Constants

### `_STOPWORDS`

Type: `frozenset`

Tokens stripped from queries and corpus text before scoring.

| Members |
|---------|
| `'a'` |
| `'an'` |
| `'the'` |
| `'how'` |
| `'do'` |
| `'does'` |
| `'i'` |
| `'to'` |
| `'with'` |
| `'for'` |
| `'is'` |
| `'are'` |
| `'of'` |
| `'in'` |
| `'on'` |
| `'at'` |
| `'and'` |
| `'or'` |
| `'but'` |
| `'can'` |
| `'should'` |
| `'would'` |
| `'will'` |
| `'be'` |
| `'been'` |
| `'by'` |
| `'my'` |
| `'me'` |
| `'we'` |
| `'it'` |
| `'this'` |
| `'that'` |
| `'these'` |
| `'those'` |

### `_STEM_SUFFIXES`

Type: `tuple`

Suffixes stripped during stemming, applied in the order listed.

| Members |
|---------|
| `'ations'` |
| `'ation'` |
| `'ators'` |
| `'ator'` |
| `'ates'` |
| `'ate'` |
| `'ings'` |
| `'ing'` |
| `'ions'` |
| `'ion'` |
| `'ies'` |
| `'ers'` |
| `'ed'` |
| `'er'` |
| `'es'` |
| `'s'` |

## Source files

- `src/attune_rag/retrieval.py`

## Tags

`retrieval`, `keyword`, `scoring`, `ranking`
