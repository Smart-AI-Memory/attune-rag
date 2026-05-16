---
type: task
name: cli-task
feature: cli
depth: task
generated_at: 2026-05-15T20:03:28.195406+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Work with the CLI

Use the CLI when you want to query the RAG pipeline or inspect corpus statistics directly from the terminal without writing code.

## Prerequisites

- Access to the project source code
- `src/attune_rag/cli.py` open for reference

## Run a query or inspect the corpus

1. **Run a RAG query.** Execute `attune-rag query` followed by your question. The command prints a grounded answer with citations drawn from the indexed corpus.

2. **Inspect corpus statistics.** Execute `attune-rag corpus-info` to display statistics about the current corpus, such as document count and index state.

## Extend or modify the CLI

1. **Identify the function to change.** The CLI has two entry points in `src/attune_rag/cli.py`:
   - `build_parser()` — constructs the `argparse.ArgumentParser` and defines subcommands and flags.
   - `main(argv)` — the top-level entry point; parses arguments and dispatches to the appropriate handler. Accepts an optional `argv` list for testing; defaults to `sys.argv` when `None`.

   Read the docstring, parameters, and return type of the function you intend to change to confirm it owns the behavior you need.

2. **Edit the function.** Open `src/attune_rag/cli.py` and make your change. To add a new subcommand, register it in `build_parser()`. To change dispatch logic or exit codes, edit `main()`.

3. **Run the CLI tests.** Execute the following command to catch regressions before they affect other developers:

   ```bash
   pytest -k "cli"
   ```

## Verify success

- `attune-rag query <your question>` returns an answer followed by source citations with no traceback.
- `attune-rag corpus-info` prints corpus statistics and exits with code `0`.
- `pytest -k "cli"` reports zero failures.

## Key files

- `src/attune_rag/cli.py`
