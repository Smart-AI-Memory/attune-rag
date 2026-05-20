---
type: task
name: provenance-task
feature: provenance
depth: task
generated_at: 2026-05-20T03:26:07.275255+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Work with provenance

Use provenance when you want to record which corpus entries grounded a RAG pipeline answer and render those citations as formatted markdown for display or auditing.

## Prerequisites

- Access to `src/attune_rag/provenance.py`
- `RetrievalHit` objects from a completed retrieval pipeline run

## Build and format a citation record

1. **Call `build_citation_record()`** to convert your `RetrievalHit` objects into a `CitationRecord`.

   ```python
   from datetime import datetime, timezone
   from attune_rag.provenance import build_citation_record

   record = build_citation_record(
       query="What is the return policy?",
       hits=retrieval_hits,
       retriever_name="bm25",
       retrieved_at=datetime.now(timezone.utc),
       excerpt_chars=200,          # optional; defaults to 200
   )
   ```

   This produces a `CitationRecord` containing the query, a tuple of `CitedSource` entries (each with a `template_path`, `category`, `score`, and optional `excerpt`), the retrieval timestamp, and the retriever name.

2. **Render the record as a markdown section** by passing the `CitationRecord` to `format_citations_markdown()`.

   ```python
   from attune_rag.provenance import format_citations_markdown

   markdown = format_citations_markdown(record, base_url="https://docs.example.com")
   print(markdown)
   ```

   Pass `base_url` to turn each `CitedSource.template_path` into an absolute link. Omit it to use relative paths.

3. **Annotate response text with claim-level citations** by calling `format_claim_citations_markdown()`. Use this step when the Anthropic Citations API has returned `ClaimCitation` objects that map character spans in the response back to specific source documents.

   ```python
   from attune_rag.provenance import format_claim_citations_markdown

   annotated = format_claim_citations_markdown(
       text=response_text,
       citations=claim_citations,   # Iterable[ClaimCitation]
       base_url="https://docs.example.com",
   )
   print(annotated)
   ```

   Each `ClaimCitation` carries a `response_span`, `document_index`, `document_title`, `cited_text`, and `cited_block_index`. The function inserts footnote-style markers into the text at the positions indicated by `response_span`.

## Verify success

- `build_citation_record()` returns a `CitationRecord` whose `hits` tuple contains one `CitedSource` per `RetrievalHit`. Inspect `record.hits` to confirm the count and scores match your retrieval results.
- `format_citations_markdown()` returns a non-empty string containing markdown. Check that each `CitedSource.template_path` appears in the output.
- `format_claim_citations_markdown()` returns the original `text` with footnote markers inserted. Confirm that the number of markers equals the number of `ClaimCitation` objects you passed in.

## Key files

- `src/attune_rag/provenance.py`
