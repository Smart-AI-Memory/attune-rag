---
type: comparison
name: cli-comparison
feature: cli
depth: comparison
generated_at: 2026-06-10T06:07:13.462959+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Comparison: CLI vs Python API for attune-rag

The `attune_rag.cli` module exposes the same retrieval pipeline through a terminal command. The Python API (`build_parser`, `main`) is also importable, so you have a real choice about which surface to use. This page helps you decide.

## Feature comparison

| Capability | CLI (`attune-rag`) | Python API (`attune_rag.cli`) |
|---|---|---|
| Run a RAG query | `attune-rag query --corpus-path <dir>` | `main(["query", "--corpus-path", "<dir>"])` |
| Select retriever | `--retriever {keyword,hybrid,transformer}` | Same flag passed as list element |
| Set abstention threshold | `--min-score <float>` | Same flag passed as list element |
| Pick prompt template | `--prompt-variant <name>` | Same flag passed as list element |
| Inspect corpus statistics | `attune-rag corpus-info --corpus-path <dir>` | `main(["corpus-info", "--corpus-path", "<dir>"])` |
| List installed LLM providers | `attune-rag providers` | `main(["providers"])` |
| Exit codes on setup errors | Exits `2` with a one-line message | Returns `2`; your code must check the return value |
| Output format | Terminal output (stdout) | Same stdout; no structured return object |
| CI/CD scripting | Yes — exit codes map directly to shell `&&` / `||` | Possible, but adds a Python subprocess wrapper for no gain |
| Interactive exploration | Yes — flags are tab-completable in most shells | No — requires writing and running a script |
| Programmatic composition | No — no return value beyond an exit code | Yes — call `main(argv)` inside larger Python workflows |
| Custom parser extension | Not without forking | Yes — call `build_parser()` and add subcommands before parsing |

## Key tradeoffs

**CLI wins for operational work.** Because `main()` exits with code `2` on setup errors (missing extras, bad paths, conflicting flags) rather than raising a traceback, the CLI is the safest surface for CI/CD pipelines and shell scripts. You get deterministic exit codes and human-readable one-line error messages with no extra error-handling code.

**Python API wins for programmatic composition.** If you need to call `main(argv)` from inside a larger Python program, pass computed flag values, or extend the argument parser by calling `build_parser()` directly, the importable API is the right choice. Be aware that `main()` writes to stdout rather than returning a structured object, so you'll need to capture stdout if you want to process the output downstream.

**They share the same flag surface.** Every flag available on the CLI (`--corpus-path`, `--retriever`, `--min-score`, `--prompt-variant`) maps directly to an `argv` element when calling `main()`. There is no capability hidden behind one surface that the other cannot reach.

## When to use each

**Use `attune-rag` from the terminal when:**
- You are running one-off queries or exploring a corpus interactively
- You are integrating retrieval into a CI/CD pipeline and need reliable exit codes
- You want error messages printed directly to the terminal without writing any error-handling code
- You are scripting batch queries in shell and chaining commands with `&&` or `||`

**Use `main(argv)` or `build_parser()` from Python when:**
- You need to call the retrieval pipeline from inside a larger Python program
- You want to compute flag values dynamically (e.g., `--corpus-path` derived from runtime config)
- You want to extend the argument parser by calling `build_parser()` and adding your own subcommands or flags before parsing
- You are writing tests that need to exercise the CLI surface without spawning a subprocess

**Do not use either when:**
- You need a structured return value rather than stdout text — neither `main()` nor the CLI returns parsed results as a Python object
- You need to run multiple concurrent queries — `main()` is a blocking call designed for single invocations

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
