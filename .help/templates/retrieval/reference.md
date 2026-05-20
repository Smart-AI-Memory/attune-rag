---
type: reference
name: retrieval-reference
feature: retrieval
depth: reference
generated_at: 2026-05-20T03:22:04.678906+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Retrieval reference

Use this module to score and rank `RetrievalEntry` objects against a natural-language query. `KeywordRetriever` tokenizes the query, strips stopwords, applies suffix stemming, and weights matches across the path, summary, content, and related fields of each entry.

## Classes

| Class | Description |
|-------|-------------|
| `RetrievalHit` | A single retrieval result. |
| `RetrieverProtocol` | Any object with a `retrieve(query, corpus, k)` method. |
| `KeywordRetriever` | Token-overlap retriever with path / summary / content / related weights. |

### `RetrievalHit`

`[dataclass]` — A single retrieval result.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `entry` | `RetrievalEntry` | |
| `score` | `float` | |
| `match_reason` | `str` | |

---

### `RetrieverProtocol`

Structural protocol for retriever objects. Any class that implements `retrieve(query, corpus, k)` with the signature below satisfies this protocol.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `self, query: str, corpus: CorpusProtocol, k: int = 3` | `Iterable[RetrievalHit]` | Retrieve the top-`k` hits from `corpus` for the given `query`. |

---

### `KeywordRetriever`

Token-overlap retriever that scores entries by weighted keyword matches across path, summary, content, and related fields.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `self, query: str, corpus: CorpusProtocol, k: int = 3` | `list[RetrievalHit]` | Retrieve the top-`k` scored `RetrievalHit` objects from `corpus` for the given `query`. |

---

## Module constants

### Stopwords

Tokens filtered from the query before scoring.

| Constant | Type | Members |
|----------|------|---------|
| `_STOPWORDS` | `frozenset` | `'a'`, `'an'`, `'the'`, `'how'`, `'do'`, `'does'`, `'i'`, `'to'`, `'with'`, `'for'`, `'is'`, `'are'`, `'of'`, `'in'`, `'on'`, `'at'`, `'and'`, `'or'`, `'but'`, `'can'`, `'should'`, `'would'`, `'will'`, `'be'`, `'been'`, `'by'`, `'my'`, `'me'`, `'we'`, `'it'`, `'this'`, `'that'`, `'these'`, `'those'` |

### Stem suffixes

Suffixes stripped during tokenization, applied in the order listed.

| Constant | Type | Members |
|----------|------|---------|
| `_STEM_SUFFIXES` | `tuple` | `'ations'`, `'ation'`, `'ators'`, `'ator'`, `'ates'`, `'ate'`, `'ings'`, `'ing'`, `'ions'`, `'ion'`, `'ies'`, `'ers'`, `'ed'`, `'er'`, `'es'`, `'s'` |

---

## Source files

- `src/attune_rag/retrieval.py`

## Tags

`retrieval`, `keyword`, `scoring`, `ranking`
