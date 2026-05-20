---
type: concept
name: cli-concept
feature: cli
depth: concept
generated_at: 2026-05-20T03:30:50.375878+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e523ece578
status: generated
---

# CLI

## Overview

The `cli` module is the command-line entry point for interacting with attune-rag, letting you run retrieval-augmented queries and inspect your corpus directly from a terminal.

Two commands are available:

- `attune-rag query` — runs a RAG query and prints a grounded answer with citations.
- `attune-rag corpus-info` — displays statistics about the loaded corpus.

This makes the CLI the primary debugging surface for retrieval behavior: you can fire a query and immediately see what the system retrieved and cited, without writing any Python.

## How the pieces fit together

Two functions form the module's structure:

- **`build_parser()`** constructs the `argparse.ArgumentParser` that defines both subcommands and their arguments. It is the single place where the CLI's interface is declared.
- **`main(argv)`** is the executable entry point. It calls `build_parser()`, parses the provided argument list (or `sys.argv` when `argv` is `None`), dispatches to the appropriate subcommand, and returns an integer exit code.

When you run `attune-rag` from the shell, your terminal invokes `main()`. Everything else — argument validation, subcommand routing, and output — flows from there.

## When this matters

The CLI is most useful when you want to:

- **Verify retrieval quality** — run a query and check whether the returned citations match what you expect.
- **Inspect corpus state** — use `corpus-info` to confirm that documents were indexed correctly before running queries.
- **Integrate with scripts** — because `main()` returns an integer exit code, you can call it from shell scripts or CI pipelines and branch on success or failure.
