---
type: concept
feature: corpus
depth: concept
generated_at: 2026-04-23T03:33:50.121496+00:00
source_hash: fbf3871db9ff126e132e66618572aafec8f5d3bb4da48be33dbd1beb2a75d455
status: generated
---

# Corpus

## What it is

A corpus is a structured collection of documents that the RAG system can retrieve from. Each document becomes a `RetrievalEntry` with standardized fields for content, metadata, and relationships.

The corpus system uses a pluggable loader architecture. You define what documents to include through a `CorpusProtocol` implementation, and the retrieval engine handles the rest.

## Core components

**`RetrievalEntry`** represents a single document in the corpus. Each entry contains the document's file path, content category, full text, an optional summary, related document paths, and arbitrary metadata. This standardized shape lets different loaders produce consistent output.

**`CorpusProtocol`** defines the interface all corpus loaders must implement. Any object that can iterate over `RetrievalEntry` objects and retrieve entries by path qualifies as a corpus. This abstraction lets you swap data sources without changing retrieval logic.

**`DirectoryCorpus`** loads markdown files from a local directory. It walks the file tree using a configurable glob pattern, parses each markdown file into a `RetrievalEntry`, and optionally loads summaries and cross-references from sidecar files.

**`AttuneHelpCorpus`** provides access to the bundled help templates. This gives you a working corpus out of the box, with professionally authored examples of different template types and documentation patterns.

## How the pieces connect

The corpus system sits between your documents and the retrieval engine. When you configure retrieval, you specify a corpus implementation. The engine calls `entries()` to discover all available documents and `get()` to fetch specific ones by path.

Different corpus types serve different use cases. Use `DirectoryCorpus` when your documents live in version control alongside your code. Use `AttuneHelpCorpus` when you want immediate access to template examples and style guides. The protocol design makes it straightforward to implement custom loaders for databases, APIs, or other document sources.

Both implementations support caching to avoid re-parsing files on repeated access, and they generate consistent metadata that downstream components can rely on for filtering and organization.
