---
type: warning
name: cli-warning
feature: cli
depth: warning
generated_at: 2026-05-20T03:30:50.396111+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI cautions

## What to watch for

The `cli` module is the command-line entry point for debugging retrieval. It exposes two commands: `attune-rag query`, which runs a RAG query and prints a grounded answer with citations, and `attune-rag corpus-info`, which displays corpus statistics.

Because this module bridges user input and the retrieval pipeline, mistakes here can silently produce incorrect output or swallow errors without a non-zero exit code.

## Risk areas

**`main()` exit code masking**
`main(argv)` returns an `int` exit code, but callers that ignore the return value — including shell wrappers or test harnesses that don't assert on it — will miss failure signals. A failed retrieval or misconfigured corpus may print partial output and still return `0` if error handling is incomplete. Always assert on the return value in tests and propagate it to the shell via `sys.exit(main())`.

**`build_parser()` default argument behavior**
`build_parser()` constructs the argument parser with its own defaults. If you call `main(argv=None)`, it reads from `sys.argv[1:]`. Passing an empty list (`[]`) is not the same as passing `None` — an empty list bypasses `sys.argv` entirely and will likely trigger a parser error or use unexpected defaults. Be explicit about which argv you intend in tests.

**Unvalidated corpus path at parse time**
Argument parsing succeeds before any corpus files are opened. If a user supplies an invalid path or a missing corpus, the error surfaces later in the retrieval pipeline, not at the CLI boundary. This can produce confusing error messages that point into library internals rather than the bad argument.

## How to avoid problems

1. **Assert on the return value of `main()`.** In tests, write `assert main(["query", ...]) == 0` rather than calling `main()` as a void function. In scripts, use `sys.exit(main())` so the shell sees failures.

2. **Pass explicit argv in tests.** Always pass a fully constructed `argv` list to `main()` in tests rather than relying on `None`. This keeps tests hermetic and prevents `sys.argv` from leaking between test cases.

3. **Validate corpus inputs before invoking the CLI in automation.** If you are driving `attune-rag` programmatically, confirm that corpus paths exist and are readable before calling `main()`. This surfaces configuration errors with a clear message rather than a traceback from deep in the retrieval stack.

4. **Treat `build_parser()` as an internal detail.** The parser structure can change as commands are added or renamed. If you depend on the parser object directly — for example, to extract subcommand names — expect that to break during refactors. Prefer driving the CLI through `main(argv)`.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`
