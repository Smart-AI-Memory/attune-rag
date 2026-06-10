---
type: error
name: cli-error
feature: cli
depth: error
generated_at: 2026-06-10T06:07:13.440779+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI errors

This page covers failures that occur when invoking `attune-rag` from the command line — including bad argument combinations, missing extras, and unresolvable paths — and explains how to trace each failure back to its cause in `main()` or `build_parser()`.

## Common error signatures

Most CLI failures produce a one-line message and exit with code `2` rather than a traceback. The table below maps common symptoms to their likely cause.

| Symptom | Likely cause |
|---|---|
| `exit 2` with a usage line | `build_parser()` rejected an unknown or conflicting flag |
| `exit 2` with a one-line actionable message | `main()` caught a setup error such as a missing extras package or a bad `--corpus-path` value |
| `FileNotFoundError` or `NotADirectoryError` | The path passed to `--corpus-path` does not exist or is not a directory |
| Import error mentioning an extras package | A retriever selected with `--retriever {keyword,hybrid,transformer}` requires an extras install that is not present |
| No output, exit `0` | `--min-score` threshold caused all results to be filtered; not a crash, but worth checking the abstention threshold |

## Where errors originate

Both public functions in `attune_rag.cli` can produce failures:

- **`build_parser()`** — raises when argument parsing fails (unknown flags, conflicting options). argparse writes the error to stderr and calls `sys.exit(2)` before `main()` is reached.
- **`main(argv)`** — handles the remaining failure surface: path validation for `--corpus-path`, extras availability for the selected `--retriever`, and provider detection for `attune-rag providers`. Setup errors here print a one-line message and exit `2`; unexpected exceptions may produce a traceback.

## How to diagnose

1. **Read the exit code.** Exit `2` always means argument or setup validation failed. Any other non-zero code means an unexpected exception escaped `main()`.

2. **Check the stderr message before looking at tracebacks.** The CLI is designed to emit a one-line actionable message for known failure modes. If stderr contains such a message, resolve it directly — no traceback inspection needed.

3. **Validate `--corpus-path` independently.** Run `attune-rag corpus-info --corpus-path <your-path>` to confirm the path is readable before adding `--retriever` or `--min-score` options. This isolates path errors from retriever errors.

4. **Verify extras for the chosen retriever.** Run `attune-rag providers` to see which LLM providers and retriever backends are detected as installed. If your `--retriever` choice does not appear, install the corresponding extras package.

5. **Reproduce with a minimal invocation.** Strip the command down to `attune-rag query --corpus-path <path> "<question>"` with no optional flags. Add flags back one at a time to identify which argument triggers the failure.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
