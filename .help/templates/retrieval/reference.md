---
type: reference
feature: retrieval
depth: reference
generated_at: 2026-04-23T03:33:40.949462+00:00
source_hash: 7143f387f3dccfded707adcfa52af1fdc50a71361e9de5a4bd466bc191c3f35b
status: generated
---

# Retrieval reference

Find and rank template matches using keyword overlap scoring.

## Classes

| Class | Description |
|-------|-------------|
| `RetrievalHit` | A single retrieval result with score and match reason |
| `RetrieverProtocol` | Interface for any retrieval implementation |
| `KeywordRetriever` | Token-overlap retriever with weighted scoring for different content areas |

## RetrievalHit

| Field | Type | Description |
|-------|------|-------------|
| `entry` | `RetrievalEntry` | The matched template entry |
| `score` | `float` | Relevance score for this match |
| `match_reason` | `str` | Explanation of why this entry matched |

## RetrieverProtocol

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `query: str, corpus: CorpusProtocol, k: int = 3` | `Iterable[RetrievalHit]` | Find and rank the top k matches for the query |

## KeywordRetriever

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `retrieve` | `query: str, corpus: CorpusProtocol, k: int = 3` | `list[RetrievalHit]` | Find matches using token overlap with weighted scoring |

## Constants

| Constant | Values | Description |
|----------|--------|-------------|
| `_STOPWORDS` | `{'a', 'an', 'the', 'how', 'do', 'does', 'i', 'to', 'with', 'for', 'is', 'are', 'of', 'in', 'on', 'at', 'and', 'or', 'but', 'can', 'should', 'would', 'will', 'be', 'been', 'by', 'my', 'me', 'we', 'it', 'this', 'that', 'these', 'those'}` | Common words ignored during keyword matching |
| `_STEM_SUFFIXES` | `{'ations', 'ation', 'ators', 'ator', 'ates', 'ate', 'ings', 'ing', 'ions', 'ion', 'ies', 'ers', 'ed', 'er', 'es', 's'}` | Word endings stripped to find root forms |

## Source files

- `src/attune_rag/retrieval.py`

## Tags

`retrieval`, `keyword`, `scoring`, `ranking`
