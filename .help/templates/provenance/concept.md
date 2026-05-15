---
type: concept
name: provenance-concept
feature: provenance
depth: concept
generated_at: 2026-05-15T20:02:41.238560+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Provenance

Provenance is the RAG pipeline's answer to "where did this come from?" — it records which corpus entries grounded each response and renders that attribution as readable markdown.

## How the pieces fit together

When the pipeline answers a query, it produces two kinds of attribution data that work at different levels of granularity:

**Run-level provenance** captures everything about a single retrieval: the original query, the timestamp, the retriever that ran, and the ranked set of sources that contributed. `CitationRecord` is the container for this data. Each entry in its `hits` tuple is a `CitedSource` — a pointer to a specific template file, with its relevance score, category, and an optional excerpt of up to 200 characters.

**Claim-level provenance** goes finer. When the Anthropic Citations API attributes a specific span of the generated response to a specific passage, that link is captured in a `ClaimCitation`. Each `ClaimCitation` records the character offsets of the claim in the response (`response_span`), the index and title of the source document, the exact text that was cited, and which block within that document the citation came from.

The two rendering functions turn these data structures into output:

- `format_citations_markdown` takes a `CitationRecord` and renders the full source list as a markdown section — useful for appending a "Sources" block to any response.
- `format_claim_citations_markdown` takes the response text alongside a sequence of `ClaimCitation` objects and produces footnote-style inline attribution, linking each claim back to its source passage.

`build_citation_record` is the entry point that bridges raw retrieval results and the structured record. It converts `RetrievalHit` objects from the retriever into a fully populated `CitationRecord`, trimming each excerpt to the configured character limit.

## When provenance matters

Provenance matters any time a user or downstream system needs to verify or audit an answer:

- **Trust** — A response with rendered citations lets readers follow the chain from claim to source document.
- **Debugging** — `CitationRecord.retriever_name` and `retrieved_at` tell you exactly which retriever ran and when, making retrieval regressions easier to isolate.
- **Claim traceability** — Claim-level citations narrow attribution to the block level inside a document, not just the document itself.
