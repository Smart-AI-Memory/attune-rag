---
type: concept
feature: cli
depth: concept
generated_at: 2026-04-23T03:36:55.162724+00:00
source_hash: dd67ed58271857e52c84068665bf3e4f498258f5607603a6e6df7dac8dfc63fe
status: generated
---

# CLI

## What it is

The CLI is attune-rag's command-line interface for debugging retrieval operations directly from the terminal.

## Core commands

The CLI provides two main commands:

- **`attune-rag query`** — Runs a RAG query and prints the grounded answer with citations
- **`attune-rag corpus-info`** — Shows statistics about the current corpus

## How it works

The CLI module acts as the command-line entry point for debugging retrieval without running the full application. When you invoke `attune-rag`, the system:

1. **Parses arguments** using `build_parser()` to set up the command structure
2. **Routes commands** through `main()` to execute the requested operation
3. **Returns results** directly to stdout for immediate inspection

This design lets you test retrieval behavior, inspect corpus contents, and debug query responses from any terminal session.

## When to use it

Use the CLI when you need to:
- Test how the RAG system responds to specific queries
- Check corpus statistics during development
- Debug retrieval issues without launching the full application
- Verify that your corpus is properly indexed and accessible
