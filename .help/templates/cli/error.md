---
type: error
name: cli-error
feature: cli
depth: error
generated_at: 2026-05-20T03:30:50.390105+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e523ece578
status: generated
---

# CLI errors

## Common error signatures

Errors in the CLI fall into two categories: argument parsing failures raised before any retrieval logic runs, and runtime failures returned as non-zero exit codes from `main()`.

- **Unrecognized or missing arguments** — `argparse` prints a usage message and exits with code `2` when you pass an unknown flag or omit a required positional argument.
- **Invalid subcommand** — calling `attune-rag` without a subcommand (`query` or `corpus-info`) produces an argument error and exits before any retrieval work begins.
- **Runtime failure in `main()`** — exceptions caught inside `main()` are reflected in the integer return value; a non-zero return indicates the command did not complete successfully.

## Where errors originate

Both functions in `src/attune_rag/cli.py` are potential failure sites:

- **`build_parser()`** — constructs the `ArgumentParser`. Errors here are typically `SystemExit` raised by `argparse` in response to bad input. They occur before retrieval starts.
- **`main(argv)`** — drives the full command lifecycle. Failures here can originate from downstream retrieval or corpus logic and are surfaced as the integer return value of the function.

## How to diagnose

1. **Check the exit code.** A code of `2` points to an argument parsing error in `build_parser()`. Any other non-zero code points to a failure inside `main()`.

2. **Read the `argparse` error message.** When argument parsing fails, `argparse` prints the specific problem — missing argument, unrecognized flag, or invalid value — directly to `stderr` before exiting.

3. **Pass `argv` explicitly during debugging.** Because `main()` accepts an optional `argv: list[str]` parameter, you can call it directly in a Python session with a known-good argument list to isolate whether the failure is in argument parsing or in retrieval logic.

4. **Capture the full traceback for unexpected exceptions.** If `main()` raises rather than returning a non-zero integer, the traceback will name the exact file and line. Errors originating outside `cli.py` itself indicate the CLI is propagating a failure from the retrieval layer.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`
