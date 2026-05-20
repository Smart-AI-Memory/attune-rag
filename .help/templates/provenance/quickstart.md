---
type: quickstart
name: provenance-quickstart
feature: provenance
depth: quickstart
generated_at: 2026-05-20T03:26:07.296097+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Track citations with provenance

Build a `CitationRecord` from retrieval hits and render it as markdown in four steps.

```python
from datetime import datetime, timezone
from attune_rag.provenance import build_citation_record, format_citations_markdown

record = build_citation_record(
    query="What is the return policy?",
    hits=retrieval_hits,          # iterable of RetrievalHit objects
    retriever_name="dense-v1",
    retrieved_at=datetime.now(timezone.utc),
)
print(format_citations_markdown(record))
```

## Prerequisites

- The project is cloned and installed locally.
- You have a list of `RetrievalHit` objects from a RAG pipeline run.

## Steps

1. **Build a citation record.** Call `build_citation_record()` with your query, hits, retriever name, and retrieval timestamp. Each hit becomes a `CitedSource` stored in `record.hits`. By default, excerpts are truncated to 200 characters; pass `excerpt_chars` to change that.

2. **Render the record as markdown.** Pass the record to `format_citations_markdown()`. Supply `base_url` if you want source links to resolve to a hosted URL.

   ```python
   md = format_citations_markdown(record, base_url="https://docs.example.com")
   print(md)
   ```

   Expected output (shape):
   ```
   ## Sources

   1. **policy/returns.md** (faq, score: 0.92)
      > Items may be returned within 30 days of purchase…
   ```

3. **Annotate claim-level citations.** If the Anthropic Citations API returned `ClaimCitation` objects, use `format_claim_citations_markdown()` to inline footnote markers into the response text.

   ```python
   from attune_rag.provenance import format_claim_citations_markdown

   annotated = format_claim_citations_markdown(
       text=response_text,
       citations=claim_citations,   # iterable of ClaimCitation
       base_url="https://docs.example.com",
   )
   print(annotated)
   ```

4. **Inspect the record fields directly.** The `CitationRecord` dataclass exposes `query`, `hits`, `retrieved_at`, and `retriever_name` if you need to log or serialize provenance rather than render it.

   ```python
   for source in record.hits:
       print(source.template_path, source.score, source.excerpt)
   ```

Next: Read the `CitationRecord` and `CitedSource` reference to understand how scores and excerpts are populated from your retriever's output.

**Tags:** `provenance`, `citations`, `traceability`
