---
type: task
feature: provenance
depth: task
generated_at: 2026-04-23T03:35:13.648014+00:00
source_hash: b73a1160ff46834c79ea6e86a93d74f6cf038d000d9ddef76a85565d587b7310
status: generated
---

# Work with provenance

Use provenance tracking when you need to record and display which corpus entries supported a RAG pipeline answer.

## Prerequisites

- Access to the project source code
- Understanding of RAG retrieval workflows

## Steps

1. **Create a citation record from retrieval hits.**
   Call `build_citation_record()` after your retrieval step:
   ```python
   from attune_rag.provenance import build_citation_record
   from datetime import datetime

   record = build_citation_record(
       query="your search query",
       hits=retrieval_results,
       retriever_name="your_retriever_id",
       retrieved_at=datetime.now()
   )
   ```

2. **Format citations for display.**
   Convert the citation record to markdown:
   ```python
   from attune_rag.provenance import format_citations_markdown

   markdown_output = format_citations_markdown(
       record=record,
       base_url="https://your-docs-site.com"  # optional
   )
   ```

3. **Access individual citation details.**
   Extract specific information from the record:
   ```python
   for hit in record.hits:
       print(f"Source: {hit.template_path}")
       print(f"Category: {hit.category}")
       print(f"Score: {hit.score}")
       if hit.excerpt:
           print(f"Preview: {hit.excerpt}")
   ```

4. **Test your implementation.**
   Run provenance-related tests:
   ```bash
   pytest -k "provenance"
   ```

## Verify success

Your implementation works correctly when:
- `CitationRecord` contains the expected query and retrieval metadata
- `format_citations_markdown()` returns properly formatted citation text
- Individual `CitedSource` objects contain accurate template paths, categories, and scores
