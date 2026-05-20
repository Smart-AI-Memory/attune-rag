---
type: faq
name: cli-faq
feature: cli
depth: faq
generated_at: 2026-05-20T03:30:50.400672+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI FAQ

## What is the CLI feature?

The CLI is the command-line entry point for debugging retrieval. It lets you run RAG queries and inspect your corpus without writing any Python code.

## What commands does it provide?

- `attune-rag query` — runs a RAG query and prints the grounded answer with citations.
- `attune-rag corpus-info` — shows statistics about your corpus.

## When should I use it?

Use the CLI when you want to run a quick query or inspect corpus statistics from the terminal. If you need to integrate retrieval into your own code programmatically, use the underlying Python API directly instead.

## What are the main entry points?

Both public functions live in `src/attune_rag/cli.py`:

- `build_parser()` — constructs and returns the `argparse.ArgumentParser` for all CLI commands.
- `main(argv)` — the top-level entry point; parses arguments and dispatches to the appropriate command. Pass a list of strings to `argv` to invoke it programmatically, or leave it as `None` to read from `sys.argv`.

## How do I debug a CLI problem?

Run the CLI-specific tests first:

```
pytest -k "cli" -v
```

If the tests pass but your command still fails, re-run with logging enabled and add a `logger.debug` statement at the suspected failure point to inspect the state at runtime.

## Where is the source code?

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`
