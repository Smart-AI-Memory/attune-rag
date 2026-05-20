---
type: note
name: retrieval-note
feature: retrieval
depth: note
generated_at: 2026-05-20T03:22:04.699988+00:00
source_hash: 808240403d72c9dd7f4962d5cbde040fffbec4b4befa508d3a494fb60f9fd862
status: generated
---

# Note: retrieval

## Context

The retrieval module (`src/attune_rag/retrieval.py`) provides a keyword-based retriever that scores and ranks corpus entries against a query. It defines a protocol so you can swap in a custom retriever without changing call sites.

## Content

`KeywordRetriever` uses token-overlap scoring — it tokenizes the query, strips stopwords (for example: *a*, *the*, *how*, *should*), and applies light suffix stemming (for example: *ations → ation*, *ing → *) before comparing tokens against each `RetrievalEntry`. Scores are weighted across four fields: path, summary, content, and related entries.

Each retrieval result is returned as a `RetrievalHit` dataclass with three fields:

| Field | Type | Description |
|---|---|---|
| `entry` | `RetrievalEntry` | The matched corpus entry |
| `score` | `float` | Weighted token-overlap score |
| `match_reason` | `str` | Human-readable explanation of why the entry matched |

Any object that implements `retrieve(query: str, corpus: CorpusProtocol, k: int = 3) -> Iterable[RetrievalHit]` satisfies `RetrieverProtocol`. `KeywordRetriever` is the built-in implementation; it returns a `list[RetrievalHit]` sorted by descending score, truncated to the top `k` results (default: 3).

## Source files

- `src/attune_rag/retrieval.py`

**Tags:** `retrieval`, `keyword`, `scoring`, `ranking`
