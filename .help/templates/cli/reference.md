---
type: reference
name: cli-reference
feature: cli
depth: reference
generated_at: 2026-06-07T07:13:42.573591+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI reference

Command-line entry point for debugging retrieval. Use `attune-rag query` to run a RAG query and print the grounded answer with citations, or `attune-rag corpus-info` to show corpus statistics.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `build_parser` | — | `argparse.ArgumentParser` | Constructs and returns the argument parser for the `attune-rag` CLI. |
| `main` | `argv: list[str] \| None = None` | `int` | Runs the CLI with the given argument list, or reads from `sys.argv` when `argv` is `None`. Returns an exit code. |

## Source files

- `src/attune_rag/cli.py`

## Tags

`cli`, `query`, `corpus-info`
