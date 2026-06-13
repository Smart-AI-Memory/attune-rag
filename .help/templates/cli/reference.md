---
type: reference
name: cli-reference
feature: cli
depth: reference
generated_at: 2026-06-10T06:04:14.437291+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI reference

Command-line entry point for debugging retrieval. Use `attune-rag query` to run a RAG query and print a grounded answer with citations. Pass `--corpus-path` to point at your own markdown directory, `--retriever {keyword,hybrid,transformer}` to select a retrieval strategy, `--min-score` to set the keyword abstention threshold, and `--prompt-variant` to choose a prompt template. Use `attune-rag corpus-info` to display corpus statistics (also accepts `--corpus-path`), and `attune-rag providers` to list LLM providers whose extras are installed. Setup errors — missing extras, bad paths, conflicting flags — print a one-line actionable message and exit with code `2` instead of a traceback.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `build_parser` | — | `argparse.ArgumentParser` | Constructs and returns the argument parser for the `attune-rag` CLI. |
| `main` | `argv: list[str] \| None = None` | `int` | Runs the CLI with the given argument list, or reads from `sys.argv` when `argv` is `None`. Returns an exit code. |

## Source files

- `src/attune_rag/cli.py`

## Tags

`cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
