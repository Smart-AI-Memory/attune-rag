---
type: reference
name: cli-reference
feature: cli
depth: reference
generated_at: 2026-05-20T03:30:50.385753+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI reference

Use this module to invoke retrieval debugging from the command line. The `attune-rag` entry point supports two subcommands: `query` runs a RAG query and prints the grounded answer with citations; `corpus-info` shows corpus statistics.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `build_parser` | — | `argparse.ArgumentParser` | Constructs and returns the argument parser for the `attune-rag` command. |
| `main` | `argv: list[str] \| None = None` | `int` | Parses arguments and dispatches to the appropriate subcommand; returns an exit code. |

## Source files

- `src/attune_rag/cli.py`

## Tags

`cli`, `query`, `corpus-info`
