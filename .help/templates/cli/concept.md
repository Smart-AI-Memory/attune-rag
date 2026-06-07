---
type: concept
name: cli-concept
feature: cli
depth: concept
generated_at: 2026-06-07T07:13:42.562728+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI

The `attune_rag.cli` module is the command-line entry point for debugging retrieval — it lets you run RAG queries and inspect corpus statistics without writing any Python.

## How the pieces fit together

Two public functions define the entire interface:

- **`build_parser()`** constructs the `argparse.ArgumentParser` that defines available subcommands and their flags. Separating parser construction from execution makes it straightforward to inspect or extend the argument structure in tests and tooling.
- **`main(argv)`** wires the parser to actual execution. Pass a list of strings to drive it programmatically, or omit `argv` (defaulting to `None`) to read from the real command line.

At runtime, `main()` calls `build_parser()` internally, so the two functions represent a single pipeline: parse → dispatch → output.

## When this matters

You interact with this module whenever you need to:

- **Debug retrieval results** — run a query end-to-end from the shell and see the grounded answer with citations, without setting up a Python session.
- **Inspect corpus state** — check corpus statistics to verify that documents were indexed as expected before troubleshooting a query.

Because `main()` accepts an explicit `argv` list, you can also invoke the CLI from integration tests or scripts with full control over arguments, without spawning a subprocess.

## Entry point

`attune_rag.cli.main` is registered as the console script entry point for the package. Calling `attune-rag` on the command line resolves to this function.
