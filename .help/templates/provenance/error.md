---
type: error
name: provenance-error
feature: provenance
depth: error
generated_at: 2026-05-20T03:26:07.283795+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Provenance errors

## Common error signatures

Errors in the provenance module typically occur when building, validating, or rendering citation records. Watch for these failure patterns:

- **`TypeError` in `build_citation_record()`** — A `RetrievalHit` object is missing an expected attribute (such as `score`, `template_path`, or `category`), or `hits` is not iterable. This produces a record with incomplete `CitedSource` entries or fails before the `CitationRecord` is constructed.
- **`AttributeError` in `format_citations_markdown()`** — The `record` argument is not a valid `CitationRecord` instance, or one of its `CitedSource` entries has a `None` value where a string field (`template_path`, `category`) is required.
- **`TypeError` in `format_claim_citations_markdown()`** — The `citations` argument is not iterable, or a `ClaimCitation` entry has a malformed `response_span` (for example, a tuple with fewer than two integers).
- **Invalid `base_url`** — Passing a malformed string as `base_url` to either `format_citations_markdown()` or `format_claim_citations_markdown()` can produce broken links in the rendered markdown without raising an immediate exception.

## Where errors originate

All three public functions in `src/attune_rag/provenance.py` are potential raise sites:

- **`build_citation_record(query, hits, retriever_name, retrieved_at, excerpt_chars=200)`** — Converts raw `RetrievalHit` objects into a `CitationRecord`. Failures here mean no citation data is available downstream. Check that every hit exposes the fields that `CitedSource` requires: `template_path`, `category`, and `score`.
- **`format_citations_markdown(record, base_url=None)`** — Renders a `CitationRecord` as a markdown section. Fails if `record.hits` is malformed or if individual `CitedSource` fields are `None`.
- **`format_claim_citations_markdown(text, citations, base_url=None)`** — Annotates response text with footnote-style citations derived from `ClaimCitation` objects. Fails if `citations` contains entries whose `response_span` tuples don't index correctly into `text`.

## How to diagnose

1. **Identify which function raised.** The traceback will point to one of the three functions above. A failure in `build_citation_record()` means the `CitationRecord` was never valid; a failure in a `format_*` function means construction succeeded but the data couldn't be rendered.

2. **Inspect the `CitationRecord` and its `hits`.** After calling `build_citation_record()`, confirm that `record.hits` is a non-empty tuple and that each `CitedSource` has a non-`None` `template_path`, `category`, and a numeric `score`. An `excerpt` of `None` is acceptable — all other fields are required.

3. **Validate `ClaimCitation` spans against the response text.** If the error is in `format_claim_citations_markdown()`, check that each `ClaimCitation.response_span` is a `(start, end)` tuple where both indices are within the bounds of `text`. A span that exceeds `len(text)` will produce an index error.

4. **Confirm `retrieved_at` is a `datetime` object.** `build_citation_record()` stores `retrieved_at` directly on `CitationRecord`. Passing a string or `None` instead of a `datetime` instance will cause failures anywhere that field is serialised or compared.

## Source files

- `src/attune_rag/provenance.py`

**Tags:** `provenance`, `citations`, `traceability`
