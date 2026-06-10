---
type: troubleshooting
name: cli-troubleshooting
feature: cli
depth: troubleshooting
generated_at: 2026-06-10T06:07:13.450859+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Troubleshoot cli

## Before you start

The `attune_rag.cli` module is the command-line entry point for debugging retrieval. Its two public functions are `build_parser()`, which constructs the argument parser, and `main()`, which runs the CLI and returns an exit code. Setup errors — such as missing extras, bad paths, or conflicting flags — print a one-line actionable message and exit with code `2` instead of raising a traceback.

## Symptom table

| If you observe | Check |
|----------------|-------|
| Exit code `2` with a one-line error message | Your flags and paths — a missing extra, an invalid `--corpus-path`, or a conflicting flag combination is the most common cause |
| A Python traceback instead of a one-line error | An unhandled exception in `main()`; note the file and line the traceback names |
| The command exits `0` but produces no output or wrong output | Early returns or conditional branches in `main()` — run with a minimal set of arguments to isolate which branch executes |
| Behavior changes between runs with no code change | Environment drift: check for changed env vars, stale caches, or a recently upgraded dependency |

## Diagnosis steps

Work through these checks in order — each one is cheaper than the next.

1. **Reproduce with the smallest possible invocation.**
   Strip the command down to required arguments only and confirm the failure still occurs. This rules out flag interactions before you look at code.

2. **Check the exit code explicitly.**
   `main()` returns an `int`. If you are calling it programmatically, print or assert the return value — a silent `2` can look like success in some shells unless you inspect `$?`.

3. **Enable debug-level logging.**
   If your environment sets a log level, raise it to `DEBUG` and re-run. Log output often identifies the offending input or state without requiring code changes.

4. **Trace through `build_parser()` and `main()`.**
   Open `src/attune_rag/cli.py` and follow the path your arguments take through `build_parser()` first (argument definitions, defaults, required flags), then through `main()` (the dispatch logic that runs after parsing).

5. **Run the CLI test suite.**
   ```bash
   pytest -k "cli" -v
   ```
   If an existing test covers the failing path, its fixtures show you the expected inputs and outputs. A newly failing test also tells you when a dependency change broke behavior.

## Common fixes

**Wrong or missing `argv` passed to `main()`**
`main()` accepts `argv: list[str] | None`. When `argv` is `None`, it reads from `sys.argv[1:]`. If you are calling `main()` programmatically and passing an explicit list, confirm the list does not include the program name as its first element — that is `sys.argv[0]` territory, not `argv[1:]`.

```python
# correct
from attune_rag.cli import main
exit_code = main(["query", "--corpus-path", "./docs", "--retriever", "hybrid"])

# incorrect — passes the program name as the first argument
exit_code = main(["attune-rag", "query", ...])
```

**Missing optional extras causing exit `2`**
If the one-line error message mentions a missing provider or retriever, the corresponding extras package is not installed. Install the relevant extra:
```bash
pip install "attune-rag[<extra-name>]"
```
Then re-run `main()` or the CLI command.

**Stale environment after a dependency upgrade**
If the CLI worked previously and now exits unexpectedly, check whether a recent `pip install --upgrade` changed a dependency:
```bash
pip show <dependency-name>
```
Pin or roll back the dependency version if the new behavior is a regression.

**Parser constructed but never called**
`build_parser()` only builds the `argparse.ArgumentParser`; it does not parse arguments or run any logic. If you call `build_parser()` directly in tests or tooling, you must call `.parse_args()` on the result yourself before any CLI behavior executes.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
