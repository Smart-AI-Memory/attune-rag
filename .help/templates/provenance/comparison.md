---
type: comparison
name: provenance-comparison
feature: provenance
depth: comparison
generated_at: 2026-05-20T03:26:07.302696+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Comparison: Provenance approaches

## Context

The `provenance` module gives you two distinct ways to attribute an answer back to its source documents:

- **Pipeline-level provenance** — `CitationRecord` and `CitedSource` capture which corpus entries grounded a full RAG response, along with retrieval metadata (query, retriever name, timestamp, relevance score).
- **Claim-level provenance** — `ClaimCitation` maps individual spans of the response text back to specific document blocks, using data from the Anthropic Citations API.

Both paths share a common rendering layer (`format_citations_markdown` and `format_claim_citations_markdown`) that produces ready-to-display markdown.

## Feature comparison

| | Pipeline-level (`CitationRecord`) | Claim-level (`ClaimCitation`) |
|---|---|---|
| **Granularity** | Per retrieved document | Per response span → document block |
| **Primary data source** | `RetrievalHit` objects from your retriever | Anthropic Citations API response |
| **Entry point** | `build_citation_record()` | Consume `ClaimCitation` objects directly |
| **What it tracks** | Query, retriever name, timestamp, score, optional excerpt | Response span `(start, end)`, document index, cited text, block index |
| **Rendering function** | `format_citations_markdown(record, base_url)` | `format_claim_citations_markdown(text, citations, base_url)` |
| **Output style** | Markdown section listing all cited sources | Response text with inline footnote-style references |
| **Requires Anthropic Citations API** | No | Yes |
| **Captures retrieval score** | Yes (`CitedSource.score`) | No |
| **Captures retrieval timestamp** | Yes (`CitationRecord.retrieved_at`) | No |

## Tradeoffs

**Pipeline-level provenance** is the right default for most RAG applications. `build_citation_record()` converts your existing `RetrievalHit` objects into a structured `CitationRecord` with minimal wiring. You get a complete audit trail — what was queried, which retriever ran, when retrieval happened, and how each source scored — all rendered to markdown via `format_citations_markdown()`. The weakness is granularity: citations point to whole documents (or short excerpts up to `excerpt_chars` characters), not to the exact sentence that supported a claim.

**Claim-level provenance** is more precise but narrower in scope. `ClaimCitation` pinpoints which character span of the model's response corresponds to which block in which document, making it suitable for applications where readers need to verify individual assertions. The tradeoff is a hard dependency on the Anthropic Citations API — you cannot produce `ClaimCitation` objects from arbitrary retrievers — and you lose retrieval metadata like scores and timestamps.

## Use X when...

**Use `CitationRecord` / pipeline-level provenance when:**
- You want to audit or log the full retrieval run (who retrieved what, when, and how relevant it scored).
- Your retriever is not the Anthropic Citations API, or you need retriever-agnostic citation tracking.
- You need a source list rendered as a markdown section appended to an answer.
- You are building a new feature and want the simplest path to traceable answers.

**Use `ClaimCitation` / claim-level provenance when:**
- You need to show users exactly which sentence in a document supports each claim in the response.
- You are already using the Anthropic Citations API and have span data available.
- Auditability of individual assertions matters more than capturing retrieval metadata.

When in doubt, start with `build_citation_record()` and `format_citations_markdown()`. They cover the common case and require no external API dependency.

## Source files

- `src/attune_rag/provenance.py`

**Tags:** `provenance`, `citations`, `traceability`
