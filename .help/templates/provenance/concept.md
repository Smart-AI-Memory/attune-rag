---
type: concept
feature: provenance
depth: concept
generated_at: 2026-04-23T03:34:59.441010+00:00
source_hash: b73a1160ff46834c79ea6e86a93d74f6cf038d000d9ddef76a85565d587b7310
status: generated
---

# Provenance

## What it is

Provenance tracks which source documents contributed to each AI-generated answer, creating an auditable paper trail from user question to final response.

When you ask a question, the RAG pipeline searches the knowledge corpus and retrieves relevant documents. The provenance system captures exactly which documents were found, how relevant they were (scored 0-1), and when the search happened. This information gets packaged into citation records that can be displayed as formatted references.

## Why it matters

AI answers without citations are black boxes. You can't verify claims, trace errors back to their source, or understand why the system gave a particular response. Provenance solves this by making every answer accountable to its source material.

Consider debugging a workflow issue. The AI suggests checking a configuration file, citing three specific documentation sections. With provenance, you can:
- Verify the AI interpreted those sections correctly
- Check if the source docs are current
- Report inaccuracies back to specific corpus entries
- Understand why certain solutions were recommended over others

## Core components

**CitedSource** represents a single document that contributed to an answer. Each source includes its file path, category (like "concept" or "task"), relevance score, and an optional excerpt showing the specific text that was relevant.

**CitationRecord** bundles all sources for one query-response cycle. It captures the original question, all retrieved sources, the search timestamp, and which retriever algorithm was used. This creates a complete audit trail for each interaction.

The formatting function renders citation records as markdown, turning internal data structures into readable reference lists that users can follow up on.

## Data flow

1. User asks a question
2. RAG retriever searches the corpus and scores matches
3. `build_citation_record()` converts search results into a structured record
4. The AI generates an answer using the retrieved sources
5. `format_citations_markdown()` renders the citation record as a reference list
6. User sees both the answer and its supporting sources

This pipeline ensures every AI response can be traced back to its evidence, making the system transparent and debuggable.
