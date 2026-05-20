---
type: troubleshooting
name: provenance-troubleshooting
feature: provenance
depth: troubleshooting
generated_at: 2026-05-20T03:26:07.291418+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Troubleshoot provenance

## Before you start

The provenance module (`src/attune_rag/provenance.py`) records which corpus entries grounded each answer and formats that record for display. The core data flow is:

1. `build_citation_record()` converts `RetrievalHit` objects into a `CitationRecord` (with `CitedSource` entries).
2. `format_citations_markdown()` renders a `CitationRecord` as a markdown section.
3. `format_claim_citations_markdown()` annotates response text with footnote-style citations from the Anthropic Citations API (`ClaimCitation` objects).

Keep this flow in mind as you work through the steps below.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `format_citations_markdown()` returns empty or malformed markdown | Confirm the `CitationRecord` passed in has a non-empty `hits` tuple and that each `CitedSource.score` is a valid `float` |
| `format_claim_citations_markdown()` produces no footnotes | Verify the `citations` iterable is not empty and that each `ClaimCitation.response_span` falls within the bounds of `text` |
| `build_citation_record()` raises an exception | Check that every object in `hits` exposes the attributes `build_citation_record()` expects from a `RetrievalHit`; a duck-typing mismatch is the most common cause |
| `CitedSource.excerpt` is `None` when you expect text | `build_citation_record()` truncates excerpts to `excerpt_chars` (default 200); pass a larger value if the source text is being cut to zero |
| Citations reference the wrong document | Inspect `ClaimCitation.document_index` against the ordered list of documents you passed to the Citations API — the index is zero-based |
| Markdown links are missing or broken | Confirm you are passing a non-`None` `base_url` to `format_citations_markdown()` or `format_claim_citations_markdown()`; without it, links are not rendered |

## Step-by-step diagnosis

Work through these steps in order — each one is cheaper than the next.

1. **Reproduce the failure in isolation.**
   Strip the call down to its required arguments and confirm the failure still occurs outside the surrounding application context. For example:

   ```python
   from datetime import datetime, timezone
   from attune_rag.provenance import build_citation_record, format_citations_markdown

   record = build_citation_record(
       query="test query",
       hits=[],          # replace with your actual hits
       retriever_name="my-retriever",
       retrieved_at=datetime.now(timezone.utc),
   )
   print(format_citations_markdown(record))
   ```

2. **Inspect the data at each stage.**
   Print (or assert on) the intermediate values before they reach the formatting functions:

   ```python
   print(record.hits)           # Are CitedSource entries present?
   print(record.retrieved_at)   # Is the datetime timezone-aware?
   for src in record.hits:
       print(src.template_path, src.score, src.excerpt)
   ```

   For claim citations, check the span and index values directly:

   ```python
   for c in citations:
       print(c.response_span, c.document_index, c.cited_text)
   ```

3. **Run the provenance tests.**

   ```bash
   pytest -k "provenance" -v
   ```

   If a test already exercises your failing path, its fixtures give you a known-good input to compare against.

4. **Enable DEBUG logging.**
   If your application configures logging for the `attune_rag` namespace, set it to `DEBUG` and re-run:

   ```python
   import logging
   logging.getLogger("attune_rag").setLevel(logging.DEBUG)
   ```

   Look for unexpected `None` values or short-circuit returns in the output.

## Common fixes

- **Empty `hits` tuple passed to `build_citation_record()`.**
  If retrieval returned no results, `CitationRecord.hits` will be an empty tuple and the formatted output will be empty. Verify your retriever is returning results before calling `build_citation_record()`.

- **`RetrievalHit` attributes missing.**
  `build_citation_record()` reads attributes from each object in `hits`. If you pass a custom object that is missing an expected attribute, you'll get an `AttributeError`. Confirm your hit objects match the expected interface.

- **Truncated excerpts.**
  The default `excerpt_chars=200` in `build_citation_record()` may cut content too short. Increase it at the call site:

  ```python
  record = build_citation_record(..., excerpt_chars=500)
  ```

- **`base_url` omitted from formatting calls.**
  Both `format_citations_markdown()` and `format_claim_citations_markdown()` accept an optional `base_url`. If you expect hyperlinked citations in the output, pass the base URL explicitly:

  ```python
  md = format_citations_markdown(record, base_url="https://docs.example.com")
  ```

- **`ClaimCitation.response_span` out of range.**
  If `response_span` indices exceed the length of `text`, annotation will silently skip or misplace the footnote. Confirm the `text` argument you pass to `format_claim_citations_markdown()` is the same string the Citations API produced the spans against.

- **Dependency version mismatch.**
  A change in the Anthropic SDK can alter the shape of Citations API responses. Run:

  ```bash
  pip show anthropic
  ```

  and confirm the installed version matches what your project requires.

## Source files

- `src/attune_rag/provenance.py`

**Tags:** `provenance`, `citations`, `traceability`
