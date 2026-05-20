---
type: concept
name: provenance-concept
feature: provenance
depth: concept
generated_at: 2026-05-20T03:26:07.269328+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Provenance

Provenance is the system that records which source documents grounded a RAG pipeline's answer and renders those sources as citable, human-readable output.

## Mental model

When a user submits a query, the RAG pipeline retrieves document chunks and generates a response. Provenance captures a snapshot of that retrieval — what was queried, which documents were returned, and how confidently each one ranked — then attaches that snapshot to the response so readers can trace every claim back to its source.

The flow looks like this:

1. `build_citation_record` converts raw `RetrievalHit` objects into a `CitationRecord`, capturing the query text, retriever name, retrieval timestamp, and up to `excerpt_chars` characters of each hit's content.
2. The `CitationRecord` holds one `CitedSource` per retrieved document, each carrying the template path, category, relevance score, and optional excerpt.
3. Optionally, the Anthropic Citations API produces `ClaimCitation` objects that link specific spans of the response text to specific documents by index.
4. `format_citations_markdown` renders a full `CitationRecord` as a markdown section. `format_claim_citations_markdown` renders the response text with inline footnote-style callouts for each `ClaimCitation`.

## Core data structures

**`CitationRecord`** — the top-level provenance snapshot for one pipeline run. It records:
- `query` — the original user query string
- `hits` — an ordered tuple of `CitedSource` objects
- `retrieved_at` — the timestamp of retrieval
- `retriever_name` — which retriever produced the hits

**`CitedSource`** — one document returned by the retriever. It records:
- `template_path` — the document's path in the corpus
- `category` — the document's classification
- `score` — the retriever's relevance score for this hit
- `excerpt` — an optional short extract of the source text (truncated to `excerpt_chars` by `build_citation_record`)

**`ClaimCitation`** — a finer-grained citation produced by the Anthropic Citations API, linking a span of the response text (`response_span`) to a specific document (`document_index`, `document_title`, `cited_text`, `cited_block_index`). Use these when you need to attribute individual sentences or phrases rather than the response as a whole.

## When provenance matters

- **Auditability** — `CitationRecord` gives you a durable, timestamped record of exactly which documents were in scope when an answer was generated, making it straightforward to replay or audit a pipeline run.
- **User-facing transparency** — `format_citations_markdown` and `format_claim_citations_markdown` turn that record into output readers can inspect, with optional `base_url` support for linking directly to source documents.
- **Claim-level traceability** — when the Anthropic Citations API is in use, `ClaimCitation` objects let you show which sentence in the response came from which block of which document, rather than citing sources only at the response level.
