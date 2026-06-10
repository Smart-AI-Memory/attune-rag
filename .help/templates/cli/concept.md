---
type: concept
name: cli-concept
feature: cli
depth: concept
generated_at: 2026-06-10T06:04:14.426036+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# The attune-rag CLI

The `attune-rag` command-line interface is the entry point for running retrieval-augmented generation queries, inspecting your corpus, and checking provider availability — without writing any Python.

## How the pieces fit together

Two functions in `attune_rag.cli` underpin everything the CLI does:

- **`build_parser()`** constructs the `argparse.ArgumentParser` that defines all subcommands and flags. Calling it directly is useful if you want to inspect available options programmatically.
- **`main(argv)`** wires the parser to your terminal session. It accepts an optional list of strings so you can invoke it from Python tests or scripts the same way you would from a shell. It returns an integer exit code.

At runtime, `main()` calls `build_parser()`, dispatches to the appropriate subcommand handler, and prints results to stdout. Setup problems — missing extras, unresolvable paths, or conflicting flags — produce a single actionable message and exit with code `2` instead of a traceback.

## Subcommands and flags

| Subcommand | What it does |
|---|---|
| `attune-rag query` | Runs a RAG query and prints the grounded answer with citations |
| `attune-rag corpus-info` | Prints statistics about the indexed corpus |
| `attune-rag providers` | Lists LLM providers whose optional extras are installed |
| `attune-rag dashboard` | Refreshes, renders, or shows the living-docs dashboard (`refresh`, `render`, `show`) |

The `query` and `corpus-info` subcommands share a common flag:

- **`--corpus-path`** — points the tool at a markdown directory of your choice instead of the default location.

The `query` subcommand also accepts:

- **`--retriever {keyword,hybrid,transformer}`** — selects a retrieval strategy. The three options form an opt-in ladder from lightweight keyword matching up to transformer-based semantic search.
- **`--min-score`** — sets the keyword abstention threshold; queries whose top result falls below this score return no answer rather than a low-confidence one.
- **`--prompt-variant`** — picks which prompt template shapes the final answer.

## When this matters

You interact with `attune_rag.cli` any time you want to:

- Test retrieval quality against a corpus without writing a Python script.
- Switch retrieval strategies (`keyword`, `hybrid`, or `transformer`) to compare results on the same query.
- Confirm which LLM provider extras are correctly installed in your environment before wiring up application code.

If you are embedding retrieval in your own code rather than running it from a shell, you can bypass the CLI entirely and call the underlying retrieval functions directly. The CLI is a convenience wrapper, not the only way in.
