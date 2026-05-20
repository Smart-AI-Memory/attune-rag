---
type: warning
name: provenance-warning
feature: provenance
depth: warning
generated_at: 2026-05-20T03:26:07.289269+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Provenance cautions

## What to watch for

The provenance module records which corpus entries grounded each answer (`CitationRecord`, `CitedSource`) and renders that provenance for display (`format_citations_markdown`, `format_claim_citations_markdown`). The risks below are specific to how these two concerns interact.

## Risk areas

### `build_citation_record` silently truncates excerpts

`build_citation_record` accepts an `excerpt_chars` parameter that defaults to `200`. If you call it without setting this value, every `CitedSource.excerpt` is capped at 200 characters. Downstream rendering via `format_citations_markdown` will then display truncated text without any indication that content was cut. Pass an explicit `excerpt_chars` value — or `None` if the retriever supports full-text excerpts — to avoid silent data loss in the citation record.

### `format_claim_citations_markdown` relies on character-offset alignment

`ClaimCitation.response_span` is a `tuple[int, int]` of character offsets into the original response text. If you modify `text` between receiving it from the model and passing it to `format_claim_citations_markdown`, the spans will no longer align and footnote markers will appear at the wrong positions or raise an index error. Pass the original, unmodified response string.

### `cited_block_index` defaults to `0`, masking multi-block documents

`ClaimCitation.cited_block_index` defaults to `0`. For documents with multiple content blocks, this default silently points every citation to the first block unless the Citations API explicitly sets a different value. If you display `cited_block_index` to users or use it for navigation, verify that the upstream API response actually populated it before trusting the value.

### `base_url=None` produces relative links that may not resolve

Both `format_citations_markdown` and `format_claim_citations_markdown` accept an optional `base_url`. When `base_url` is `None`, any links in the rendered markdown are relative and depend entirely on the serving context to resolve correctly. In standalone outputs — emails, exported PDFs, or API responses consumed outside your app — those links will be broken. Always supply an absolute `base_url` when the rendered markdown may be consumed outside a known URL context.

## How to avoid problems

1. **Pin `excerpt_chars` explicitly.** Treat the `200`-character default as a footgun rather than a sensible default. Set it deliberately every time you call `build_citation_record` so truncation behavior is visible in code review.

2. **Freeze response text before rendering citations.** Treat the string you pass to `format_claim_citations_markdown` as immutable from the moment you receive model output. Apply any sanitization or formatting to a copy, not the string you will use for citation rendering.

3. **Validate `cited_block_index` before use.** If your UI navigates users to a specific block in a source document, assert that `cited_block_index > 0` or that it matches an expected block count before rendering the link. Do not rely on the default-`0` value as confirmation that block `0` was actually cited.

4. **Supply `base_url` in non-browser contexts.** If `format_citations_markdown` output leaves your web application — for example, in a notification, a report, or an API response — pass an absolute `base_url` so citation links remain functional.

5. **Isolate provenance tests from retrieval state.** `CitationRecord` captures `retrieved_at` and `retriever_name` at construction time. Tests that reuse a shared record across assertions may be checking stale retrieval metadata. Construct a fresh `CitationRecord` per test case.

## Source files

- `src/attune_rag/provenance.py`

**Tags:** `provenance`, `citations`, `traceability`
