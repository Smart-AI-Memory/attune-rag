---
type: note
name: provenance-note
feature: provenance
depth: note
generated_at: 2026-05-20T03:26:07.300164+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Note: provenance

## Context

The provenance module (`src/attune_rag/provenance.py`) records which corpus entries grounded each RAG pipeline answer and renders that attribution as formatted markdown. It covers two levels of granularity: whole-response citations and claim-level citations tied to specific spans in the response text.

## Content

A single RAG pipeline run produces one `CitationRecord`, which holds the original query, the retriever name, the retrieval timestamp, and a tuple of `CitedSource` entries. Each `CitedSource` identifies a retrieved document by its template path and category, carries a relevance score, and optionally includes a short excerpt (truncated to `excerpt_chars`, defaulting to 200 characters) via `build_citation_record()`.

Claim-level attribution uses `ClaimCitation`, which is populated by the Anthropic Citations API. Each `ClaimCitation` maps a character span in the response (`response_span`) back to a specific document and block (`document_index`, `cited_block_index`) and records the exact text that was cited.

The two rendering functions operate at matching granularities:

- `format_citations_markdown(record, base_url)` renders a full `CitationRecord` as a markdown section, optionally linking sources to a `base_url`.
- `format_claim_citations_markdown(text, citations, base_url)` renders response text with footnote-style markers for each `ClaimCitation`.

`build_citation_record()` is the standard way to construct a `CitationRecord` from raw `RetrievalHit` objects returned by a retriever.

> **Note:** `ClaimCitation` is produced by the Anthropic Citations API and is only populated when that API is in use. `CitationRecord` and `CitedSource` are populated for all retriever backends.

## Source files

- `src/attune_rag/provenance.py`

**Tags:** `provenance`, `citations`, `traceability`
