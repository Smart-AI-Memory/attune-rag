---
type: reference
feature: provenance
depth: reference
generated_at: 2026-04-23T03:35:22.361999+00:00
source_hash: b73a1160ff46834c79ea6e86a93d74f6cf038d000d9ddef76a85565d587b7310
status: generated
---

# Provenance reference

Record and format citation data for RAG pipeline traceability. Track query provenance with source attribution and render citations as markdown.

## Classes

| Class | Description |
|-------|-------------|
| `CitedSource` | A single cited source within a CitationRecord |
| `CitationRecord` | Provenance for a single RAG pipeline run |

### CitedSource

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `template_path` | `str` | | Path to the source template |
| `category` | `str` | | Classification of the source type |
| `score` | `float` | | Relevance score from retrieval |
| `excerpt` | `str \| None` | `None` | Optional text snippet from the source |

### CitationRecord

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `str` | | The original user query |
| `hits` | `tuple[CitedSource, ...]` | | Retrieved sources for this query |
| `retrieved_at` | `datetime` | | Timestamp of the retrieval |
| `retriever_name` | `str` | | Identifier of the retrieval system used |

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `format_citations_markdown` | `record: CitationRecord`, `base_url: str \| None = None` | `str` | Render a CitationRecord as a markdown section |
| `build_citation_record` | `query: str`, `hits: Iterable`, `retriever_name: str`, `retrieved_at: datetime`, `excerpt_chars: int = 200` | `CitationRecord` | Convert RetrievalHit objects into a CitationRecord |
