---
type: tip
name: cli-tip
feature: cli
depth: tip
generated_at: 2026-05-20T03:30:50.405218+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Tip: working effectively with cli

## Recommendation

Pass a pre-built argument list to `main(argv)` instead of relying on `sys.argv` when you call the CLI programmatically.

`main()` accepts an explicit `argv: list[str] | None` parameter. When `argv` is `None`, it falls back to `sys.argv`, which makes behavior depend on the process environment. Passing a concrete list — for example, `main(["query", "--question", "What is RAG?"])` — makes the call self-contained and predictable.

**Why it matters:** Scripts and tests that mutate `sys.argv` are fragile and interfere with each other; an explicit `argv` eliminates that coupling entirely.

**Tradeoff:** You need to reproduce the argument structure that `build_parser()` expects. Call `build_parser().parse_args(your_list)` first to validate your argument list before wiring it into `main()`.

## Source files

- `src/attune_rag/cli.py`

**Tags:** `cli`, `query`, `corpus-info`
