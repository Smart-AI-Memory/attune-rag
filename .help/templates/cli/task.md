---
type: task
name: cli-task
feature: cli
depth: task
generated_at: 2026-06-07T07:13:42.568215+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Work with the attune-rag CLI

Use the `attune_rag.cli` module when you need to run retrieval queries or inspect corpus statistics from the command line.

## Prerequisites

- Access to the project source code at `src/attune_rag/cli.py`
- A working Python environment with `attune_rag` installed

## Steps

1. **Review the two public entry points.**
   Open `src/attune_rag/cli.py` and locate `build_parser()` and `main()`. `build_parser()` constructs the argument parser that defines all available subcommands and flags. `main()` is the entry point that parses `argv` and dispatches to the appropriate handler.

2. **Identify which function owns the behavior you need.**
   - To add or modify a subcommand, flag, or argument, work in `build_parser()`.
   - To change how the CLI dispatches, handles errors, or returns exit codes, work in `main()`.

3. **Edit the target function.**
   Make your changes in `src/attune_rag/cli.py`. Keep argument names consistent with those already defined in `build_parser()`, and ensure `main()` returns an integer exit code.

4. **Run the related tests.**
   Execute `pytest -k "cli"` to catch regressions before they affect other developers.

5. **Verify the CLI behaves as expected.**
   Call `main()` directly with a list of arguments, or invoke the installed command from your shell. Confirm the output, exit code, and any error messages match your intent.

## Verify success

You know this task is complete when:

- `pytest -k "cli"` passes with no failures
- Calling `main()` with valid arguments returns `0`
- Calling `main()` with invalid arguments returns a non-zero exit code and prints a usage message

## Key files

- `src/attune_rag/cli.py` — contains `build_parser()` and `main()`
