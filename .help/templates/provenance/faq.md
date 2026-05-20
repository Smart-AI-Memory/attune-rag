---
type: faq
name: provenance-faq
feature: provenance
depth: faq
generated_at: 2026-05-20T03:26:07.293846+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Provenance FAQ

## What is provenance?

Provenance is citation tracking for your RAG pipeline. A `CitationRecord` captures which corpus entries grounded a given answer, and the formatting functions render those citations for display.

## When should I use it?

Use provenance whenever you need to trace an answer back to its source documents — for example, to show users which templates were retrieved, or to audit retrieval quality after a pipeline run.

## What functions do I call first?

It depends on what you need:

- **Building a record:** Call `build_citation_record()` after retrieval to convert your `RetrievalHit` objects into a `CitationRecord`. This is almost always the first step.
- **Rendering source-level citations:** Call `format_citations_markdown(record, base_url)` to produce a markdown section listing every `CitedSource` in the record.
- **Rendering claim-level citations:** Call `format_claim_citations_markdown(text, citations, base_url)` to annotate response text with footnote-style citations produced by the Anthropic Citations API.

All three functions live in `src/attune_rag/provenance.py`.

## What does a `CitationRecord` contain?

A `CitationRecord` holds the original `query`, a tuple of `CitedSource` hits, a `retrieved_at` timestamp, and the `retriever_name`. Each `CitedSource` records the `template_path`, `category`, relevance `score`, and an optional `excerpt`.

## What is a `ClaimCitation` and how does it differ from a `CitedSource`?

A `CitedSource` is a document retrieved during the RAG lookup and stored in a `CitationRecord`. A `ClaimCitation` is a finer-grained citation produced by the Anthropic Citations API: it points to a specific span in the response text (`response_span`), names the source document (`document_index`, `document_title`), and quotes the exact passage (`cited_text`). Use `ClaimCitation` objects with `format_claim_citations_markdown()` when you need sentence- or claim-level attribution.

## Can I link citations to a hosted copy of the source documents?

Yes. Both `format_citations_markdown()` and `format_claim_citations_markdown()` accept an optional `base_url` argument. When provided, the rendered markdown links each citation to that base URL.

## How do I debug a provenance issue?

Run the provenance tests first:

```
pytest -k "provenance" -v
```

If the tests pass but your code still fails, add a `logger.debug` statement just after `build_citation_record()` to inspect the `CitationRecord` before it reaches the formatting functions. Check that `hits` is not empty and that each `CitedSource` has the expected `template_path` and `score`.

## Where is the source code?

Everything described here is in `src/attune_rag/provenance.py`.

**Tags:** `provenance`, `citations`, `traceability`
