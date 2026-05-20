---
type: tip
name: provenance-tip
feature: provenance
depth: tip
generated_at: 2026-05-20T03:26:07.298278+00:00
source_hash: 2ad01dedc91108386ca6445b49decedb0fa3b58762c00286b0a0e45fed8409a7
status: generated
---

# Tip: working effectively with provenance

## Recommendation

Use `build_citation_record()` to convert retrieval hits into a `CitationRecord`, then pass that record directly to `format_citations_markdown()` or `format_claim_citations_markdown()` — don't assemble `CitationRecord` or `CitedSource` dataclasses by hand.

**Why:** `build_citation_record()` handles excerpt truncation (via `excerpt_chars`) and timestamp stamping consistently; constructing the dataclasses manually skips that logic and produces records that are structurally valid but semantically incomplete.

## Tradeoff

Routing everything through `build_citation_record()` means your input must be `RetrievalHit` objects. If you're working with a custom retriever that doesn't produce `RetrievalHit`s, you'll need an adapter layer — but that's preferable to bypassing provenance tracking entirely or duplicating the excerpt-truncation logic yourself.

## Source files

- `src/attune_rag/provenance.py`

**Tags:** `provenance`, `citations`, `traceability`
