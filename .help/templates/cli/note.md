---
type: note
name: cli-note
feature: cli
depth: note
generated_at: 2026-05-20T03:30:50.407473+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Note: cli

## Context

The `cli` module (`src/attune_rag/cli.py`) is the command-line entry point for debugging retrieval. It exposes two commands:

- `attune-rag query` — runs a RAG query and prints the grounded answer with citations.
- `attune-rag corpus-info` — prints corpus statistics.

## Content

The module contains two top-level functions; nothing needs to be instantiated before calling them:

- `build_parser() -> argparse.ArgumentParser` — constructs and returns the argument parser for both subcommands.
- `main(argv: list[str] | None = None) -> int` — parses arguments and dispatches to the appropriate subcommand. When `argv` is `None`, it reads from `sys.argv`.

Because `main()` returns an integer exit code, it is suitable for use as a `console_scripts` entry point.

**Tags:** `cli`, `query`, `corpus-info`
