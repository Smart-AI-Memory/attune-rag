---
type: concept
name: cli-concept
feature: cli
depth: concept
generated_at: 2026-05-15T20:03:28.191523+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI

The `attune-rag` command-line interface is a debugging entry point that lets you run retrieval-augmented generation queries and inspect corpus state directly from your terminal.

## Two commands, two purposes

The CLI exposes two subcommands:

- **`attune-rag query`** — runs a RAG query against the corpus and prints the grounded answer with citations.
- **`attune-rag corpus-info`** — displays corpus statistics, useful for verifying that your corpus is populated and healthy before running queries.

These commands exist specifically for debugging retrieval. If a query returns unexpected results, `corpus-info` lets you check the corpus state without writing any code.

## How the pieces fit together

Two functions in `src/attune_rag/cli.py` wire everything together:

| Function | Role |
|---|---|
| `build_parser()` | Constructs the `argparse.ArgumentParser` — defines the `query` and `corpus-info` subcommands and their arguments. |
| `main(argv)` | The executable entry point. Accepts an optional argument list (defaults to `sys.argv`), delegates to the appropriate subcommand, and returns an integer exit code. |

When you run `attune-rag`, the interpreter calls `main()`. `main()` calls `build_parser()` to understand what you typed, then routes to whichever subcommand you invoked.

## When the CLI matters

Use the CLI when you want to:

- Spot-check retrieval quality without writing a Python caller.
- Confirm corpus statistics after ingestion.
- Reproduce a query failure in isolation so you can trace it through the retrieval pipeline.

For programmatic access — calling retrieval from your own code — use the underlying Python API directly rather than shelling out to `main()`.
