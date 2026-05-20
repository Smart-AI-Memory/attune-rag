---
type: task
name: cli-task
feature: cli
depth: task
generated_at: 2026-05-20T03:30:50.381478+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Run retrieval debugging from the command line

Use the `attune-rag` CLI when you need to query the RAG pipeline or inspect corpus statistics directly from your terminal without writing Python code.

## Prerequisites

- A working installation of `attune-rag` with the CLI entry point available on your `PATH`
- Access to `src/attune_rag/cli.py` if you intend to extend or modify CLI behavior

## Run a query or inspect the corpus

1. **Run a RAG query.** Pass your question as an argument to get a grounded answer with citations:

   ```
   attune-rag query "What is the retrieval strategy?"
   ```

2. **View corpus statistics.** Run the `corpus-info` subcommand to print a summary of the current corpus:

   ```
   attune-rag corpus-info
   ```

3. **Review available options.** Use `--help` to see all subcommands and flags exposed by `build_parser()`:

   ```
   attune-rag --help
   ```

## Extend the CLI

1. **Open the entry point file.**
   All CLI logic lives in `src/attune_rag/cli.py`. The two functions you will work with are:
   - `build_parser()` — constructs and returns the `argparse.ArgumentParser`, including all subcommands and flags
   - `main(argv)` — parses arguments and dispatches to the appropriate handler; returns an integer exit code

2. **Add or modify a subcommand in `build_parser()`.** Add a new subparser or argument to the parser this function returns. Match the naming conventions and help-string style of the existing subcommands.

3. **Wire up the handler in `main()`.** Add the corresponding dispatch logic so the new subcommand calls the right function and returns an explicit integer exit code (`0` for success, non-zero for failure).

4. **Run the targeted tests** to catch regressions before they affect other developers:

   ```
   pytest -k "cli"
   ```

## Verify success

- `attune-rag query` prints an answer followed by citation references and exits with code `0`.
- `attune-rag corpus-info` prints corpus statistics and exits with code `0`.
- `pytest -k "cli"` reports no failures.

## Key files

- `src/attune_rag/cli.py`
