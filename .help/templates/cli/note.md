---
type: note
name: cli-note
feature: cli
depth: note
generated_at: 2026-06-10T06:07:13.459803+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Note: cli

## Context

`attune_rag.cli` is the command-line entry point for debugging retrieval. It exposes two public functions — `build_parser()` and `main()` — rather than a class hierarchy. You call `main()` directly (or let the installed script do so); `build_parser()` returns the `argparse.ArgumentParser` if you need to inspect or extend the argument definitions programmatically.

## Design decisions

The module is intentionally function-first: there is nothing to instantiate. `build_parser` and `main` are the entire public surface (`attune_rag.cli`).

Setup errors — missing extras, bad paths, conflicting flags — print a single actionable message and exit with code `2` instead of raising a traceback. This keeps the debugging experience consistent whether you run the tool interactively or drive it from a script.

**Tags:** `cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
