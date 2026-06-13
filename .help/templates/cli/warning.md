---
type: warning
name: cli-warning
feature: cli
depth: warning
generated_at: 2026-06-10T06:07:13.448525+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI Cautions

## What to watch for

The `attune_rag.cli` module exposes two public entry points: `build_parser()` and `main()`. `main()` accepts an optional `argv` list, which means the same function runs both from the shell and from test code — behavior that diverges across those contexts is the highest-risk surface in this module.

Setup errors (missing extras, bad paths, conflicting flags) intentionally exit with code 2 and a one-line message rather than raising an exception. If you call `main()` programmatically and check only for exceptions, you will miss these failure modes.

## Risk areas

### `main()` swallows setup failures as exit codes, not exceptions

When invoked from a shell, exit code 2 is the standard signal for a usage error. When invoked programmatically — for example, in a test that calls `main(argv=[...])` — an exit code of 2 is returned as an integer, not raised. Asserting on return value rather than absence of exception is the correct pattern here.

**Mitigation:** Always assert on the integer return value of `main()` in tests. A return value of `0` signals success; anything else signals failure. Do not wrap `main()` calls in a bare `try/except` and treat no exception as a pass.

### `build_parser()` output depends on installed extras

`build_parser()` constructs the argument parser, but the available subcommands and choices (such as retriever options) reflect whichever optional extras are installed in the current environment. A parser built in a minimal environment may silently omit choices that exist in a full installation.

**Mitigation:** When testing or scripting against `build_parser()`, verify your environment has the same extras installed as your target deployment. Mismatches between development and CI environments are a common source of "works locally, fails in CI" failures here.

### Passing `argv=None` inherits the real `sys.argv`

`main(argv=None)` falls back to reading `sys.argv` directly. In test environments where `sys.argv` is not explicitly controlled, this can cause tests to pick up arguments intended for the test runner itself.

**Mitigation:** Always pass an explicit list when calling `main()` from test code: `main(argv=["query", "--corpus-path", "..."])`. Never rely on `argv=None` outside of a true shell invocation.

## How to avoid problems

1. **Control `argv` explicitly in tests.** Pass a full argument list to `main()` rather than letting it read `sys.argv`. This isolates your test from the runner's own arguments.

2. **Check the return value, not just exceptions.** `main()` signals errors through its integer return value. A successful run returns `0`; setup and usage errors return `2`. Structure your assertions accordingly.

3. **Match your extras across environments.** `build_parser()` reflects installed packages. Pin the same extras in your CI environment as in your development environment to prevent parser shape mismatches.

4. **Treat `_`-prefixed helpers as unstable.** Only `build_parser()` and `main()` are part of the public API in `attune_rag.cli`. Internal helpers can change without notice.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
