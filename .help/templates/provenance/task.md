---
type: task
name: provenance-task
feature: provenance
depth: task
generated_at: 2026-05-15T20:02:41.242311+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Work with provenance

Use provenance when you need to track which corpus entries grounded a RAG pipeline answer and render that attribution as formatted markdown for display.

## Prerequisites

- Access to the project source code
- Familiarity with `src/attune_rag/provenance.py`

## Steps

1. **Identify the function that owns the behavior you need.**
   The module exposes three functions, each with a single responsibility:
   - `build_citation_record()` — converts `RetrievalHit` objects into a `CitationRecord` that captures the query, the cited sources, the retriever name, and the retrieval timestamp.
   - `format_citations_markdown()` — renders a complete `CitationRecord` as a markdown section.
   - `format_claim_citations_markdown()` — renders response text with footnote-style citations for individual claims.

   Read the docstring, parameters, and return type of the target function to confirm it owns the behavior you need.

2. **Build the `CitationRecord`.**
   Call `build_citation_record()` after your retrieval step, passing the query string, the iterable of `RetrievalHit` objects, the retriever name, and the retrieval timestamp:

   ```python
   record = build_citation_record(
       query=query,
       hits=retrieval_hits,
       retriever_name="my-retriever",
       retrieved_at=datetime.now(tz=timezone.utc),
   )
   ```

   Each resulting `CitedSource` in `record.hits` contains the template path, category, relevance score, and an optional excerpt.

3. **Render the citations as markdown.**
   Pass the `CitationRecord` to the appropriate formatter:

   - To render a full source list, call `format_citations_markdown(record, base_url=base_url)`.
   - To annotate individual claims in a response string, call `format_claim_citations_markdown(text, citations, base_url=base_url)`, where `citations` is an iterable of `ClaimCitation` objects produced by the Anthropic Citations API.

4. **Run the provenance tests.**
   Verify your changes haven't introduced regressions:

   ```shell
   pytest -k "provenance"
   ```

## Key files

- `src/attune_rag/provenance.py`

## Verify success

The task is complete when:

- `build_citation_record()` returns a `CitationRecord` whose `hits` tuple contains one `CitedSource` entry per relevant retrieval result.
- `format_citations_markdown()` or `format_claim_citations_markdown()` returns a non-empty markdown string that references the expected sources.
- `pytest -k "provenance"` exits with no failures.
