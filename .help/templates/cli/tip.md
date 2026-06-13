---
type: tip
name: cli-tip
feature: cli
depth: tip
generated_at: 2026-06-10T06:07:13.457881+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Tip: working effectively with cli

Pass `argv` explicitly when calling `main()` from your own scripts or tests — don't rely on `sys.argv` being set correctly in your environment.

**Why:** `main(argv: list[str] | None = None)` accepts an explicit argument list, so passing `["query", "--corpus-path", "/your/docs"]` directly gives you predictable, reproducible behavior without environment side effects.

**Tradeoff:** You take on the responsibility of constructing a valid argv list. Malformed flags still cause `main()` to exit with code 2, so test your argv construction against `build_parser()` first — call `build_parser().parse_args(your_argv)` to validate the shape before passing it to `main()`.

**Tags:** `cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
