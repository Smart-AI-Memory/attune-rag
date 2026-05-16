---
type: reference
name: cli-reference
feature: cli
depth: reference
generated_at: 2026-05-15T20:03:28.198307+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e523ece578
status: generated
---

# CLI reference

Command-line entry point for debugging retrieval. Use these functions to build and invoke the argument parser for the `attune-rag` command.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `build_parser` | — | `argparse.ArgumentParser` | Constructs and returns the argument parser for the `attune-rag` CLI. |
| `main` | `argv: list[str] | None = None` | `int` | Parses arguments and runs the CLI; returns an exit code. |

## Source files

- `src/attune_rag/cli.py`

## Tags

`cli`, `query`, `corpus-info`
