---
type: troubleshooting
name: cli-troubleshooting
feature: cli
depth: troubleshooting
generated_at: 2026-05-20T03:30:50.398331+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e523eb578
status: generated
---

# Troubleshoot cli

## Before you start

The `cli` module is the command-line entry point for debugging retrieval. It exposes two commands:

- `attune-rag query` — runs a RAG query and prints the grounded answer with citations
- `attune-rag corpus-info` — prints corpus statistics

Both commands are wired through `build_parser()`, which constructs the argument parser, and `main()`, which dispatches to the appropriate subcommand and returns an integer exit code.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `attune-rag` exits with a non-zero code unexpectedly | The integer returned by `main()` — a non-zero value signals an error path was reached |
| `command not found: attune-rag` | Whether the package is installed in the active environment: `pip show attune-rag` |
| Unrecognized arguments or usage error printed to stderr | The argument definitions in `build_parser()` — confirm flag names and required positional arguments match your invocation |
| Traceback on `attune-rag query` or `attune-rag corpus-info` | The bottom frame of the traceback in `src/attune_rag/cli.py` — Python names the exact file and line |
| Query runs but prints no citations or wrong output | Whether the correct corpus path or query arguments were passed — run with `--help` to verify expected arguments |
| Intermittent failures between runs | Environment variables or cached state that `main()` reads on each invocation |

## Step-by-step diagnosis

1. **Confirm the command fails with a minimal invocation.**
   Run the simplest possible form of the failing command directly in your terminal, outside any wrapper scripts or CI environment:
   ```sh
   attune-rag query "your test query"
   attune-rag corpus-info
   ```
   This rules out issues introduced by surrounding context and confirms the failure is reproducible.

2. **Check the `--help` output.**
   If the error is argument-related, inspect the parser's expected interface before digging further:
   ```sh
   attune-rag --help
   attune-rag query --help
   attune-rag corpus-info --help
   ```
   Compare the required arguments and flag names against your invocation.

3. **Enable debug logging and re-run.**
   If the CLI respects a log-level flag or the `LOG_LEVEL` environment variable, set it to `DEBUG` before re-running:
   ```sh
   LOG_LEVEL=DEBUG attune-rag query "your test query"
   ```
   The additional output often identifies which subsystem (retrieval, corpus loading, formatting) is failing.

4. **Read the full traceback.**
   If an exception is raised, read the traceback from the bottom up. The lowest frame in `src/attune_rag/cli.py` shows exactly where `main()` failed. Frames above it show which downstream call caused the error.

5. **Run the CLI test suite.**
   Execute the tests scoped to this module to see which paths are currently passing:
   ```sh
   pytest -k "cli" -v
   ```
   If a test covers the failing path, use its fixtures to reproduce the failure in a controlled setting.

6. **Trace the exit code back through `main()`.**
   Open `src/attune_rag/cli.py` and follow the control flow in `main()` for the subcommand you are invoking. Identify every early return and the integer it returns. Add a temporary `print()` or log statement at the suspected branch to confirm which path is taken.

## Common fixes

- **Wrong or missing argument.**
  If `build_parser()` raises a usage error, your invocation is missing a required argument or uses an unrecognized flag. Check `attune-rag <subcommand> --help` and correct the call.

- **Package not installed or wrong environment.**
  If `attune-rag` is not found, install the package in the active environment:
  ```sh
  pip install -e .
  ```
  Confirm the entry point is registered:
  ```sh
  pip show attune-rag
  which attune-rag
  ```

- **Dependency version mismatch.**
  If the command worked previously but now raises an unexpected error, a dependency upgrade may have changed behaviour. Check the installed versions of key dependencies:
  ```sh
  pip show <dependency-name>
  ```
  Pin or restore the expected version in your environment, then re-run.

- **Stale environment state.**
  If the failure is intermittent and no code changed, check for environment variables that affect retrieval or corpus loading. Unset or reset suspect variables and re-run the command in a clean shell session.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`
