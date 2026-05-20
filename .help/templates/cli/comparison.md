---
type: comparison
name: cli-comparison
feature: cli
depth: comparison
generated_at: 2026-05-20T03:30:50.409990+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e523ece578
status: generated
---

# CLI vs. direct Python API

## Context

`attune_rag.cli` is the command-line entry point for debugging retrieval. It exposes two commands:

- `attune-rag query` — runs a RAG query and prints the grounded answer with citations.
- `attune-rag corpus-info` — prints corpus statistics.

The module's public surface is two functions in `src/attune_rag/cli.py`: `build_parser()`, which constructs the argument parser, and `main(argv)`, which parses arguments and runs the selected command.

## Feature comparison

| Capability | CLI (`attune-rag`) | Direct Python API |
|---|---|---|
| **Primary use case** | Ad-hoc queries and corpus inspection from a terminal | Programmatic integration in scripts, services, or notebooks |
| **Output format** | Human-readable text with citations | Structured return values you can process further |
| **Setup overhead** | None — invoke from a shell | Requires importing and wiring up the relevant modules |
| **Debugging retrieval** | First-class: designed for this workflow | Possible, but you assemble the pipeline yourself |
| **Automation / scripting** | Limited — parse stdout to get results | Native — work directly with Python objects |
| **Argument handling** | `build_parser()` / `main(argv)` | Your own argument handling |
| **Interactive exploration** | Fast for one-off checks | Better suited to notebooks or REPLs with richer introspection |

## When to use the CLI

Use `attune-rag` when you need to:

- **Inspect or debug retrieval interactively.** The CLI is purpose-built for this; you get a grounded answer and citations without writing any code.
- **Spot-check corpus health.** `attune-rag corpus-info` gives you statistics in seconds from a terminal.
- **Reproduce a retrieval result in isolation.** Because `main()` accepts an explicit `argv` list, you can also call it from a test to capture CLI-level behaviour without spawning a subprocess.

## When to use the Python API directly

Use the Python API instead of the CLI when you need to:

- **Process results programmatically.** Parsing the CLI's stdout is fragile; if downstream code needs the answer or citations as data, call the underlying API and work with its return values.
- **Integrate retrieval into a larger pipeline.** Services, scheduled jobs, or multi-step workflows should use the orchestration layer above `cli`, not shell out to it.
- **Extend or customise behaviour.** `build_parser()` and `main()` are not designed as extension points. If the CLI does not expose a flag or behaviour you need, add it at the API level rather than patching CLI internals.
- **Do exploratory, throwaway work.** A short notebook cell or script that imports the retrieval function directly is less overhead than constructing a full CLI invocation.

## Recommendation

The CLI is the right tool for **human-driven debugging and corpus inspection**. If you are writing code that consumes results or runs unattended, the Python API is the better choice. When in doubt: if you would type the command in a terminal, use the CLI; if you would call it from code, use the API.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`
