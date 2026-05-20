---
type: reference
name: provenance-reference
feature: provenance
depth: reference
generated_at: 2026-05-20T03:26:07.279977+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Provenance reference

Record which corpus entries grounded each answer and render those citations as markdown. `CitationRecord` and `CitedSource` capture per-query provenance; `ClaimCitation` tracks span-level attribution from the Anthropic Citations API; the format functions turn records into human-readable output.

## Classes

| Class | Description |
|-------|-------------|
| `ClaimCitation` | One claim-level citation produced by the Anthropic Citations API. |
| `CitedSource` | A single cited source within a `CitationRecord`. |
| `CitationRecord` | Provenance for a single RAG pipeline run. |

### `ClaimCitation` fields

| Field | Type | Default |
|-------|------|---------|
| `response_span` | `tuple[int, int]` | |
| `document_index` | `int` | |
| `document_title` | `str` | |
| `cited_text` | `str` | |
| `cited_block_index` | `int` | `0` |

### `CitedSource` fields

| Field | Type | Default |
|-------|------|---------|
| `template_path` | `str` | |
| `category` | `str` | |
| `score` | `float` | |
| `excerpt` | `str | None` | `None` |

### `CitationRecord` fields

| Field | Type | Default |
|-------|------|---------|
| `query` | `str` | |
| `hits` | `tuple[CitedSource, ...]` | |
| `retrieved_at` | `datetime` | |
| `retriever_name` | `str` | |

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `format_citations_markdown` | `record: CitationRecord, base_url: str \| None = None` | `str` | Render a `CitationRecord` as a markdown section. |
| `format_claim_citations_markdown` | `text: str, citations: Iterable[ClaimCitation], base_url: str \| None = None` | `str` | Render response text with footnote-style claim citations. |
| `build_citation_record` | `query: str, hits: Iterable, retriever_name: str, retrieved_at: datetime, excerpt_chars: int = 200` | `CitationRecord` | Convert `RetrievalHit` objects into a `CitationRecord`. |

## Source files

- `src/attune_rag/provenance.py`

## Tags

`provenance`, `citations`, `traceability`
