---
type: comparison
name: benchmark-comparison
feature: benchmark
depth: comparison
generated_at: 2026-05-20T03:30:01.602854+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Comparison: Benchmark vs alternatives

## Context

The `benchmark` module is a retrieval and optional faithfulness benchmark runner. It gates CI pipelines on configurable thresholds, accepts custom query files, and enables faithfulness scoring through the `--with-faithfulness` flag.

## Feature breakdown

| Capability | `benchmark` | Ad-hoc evaluation script | Orchestration layer |
|---|---|---|---|
| Precision/recall measurement | ✅ Built-in | Manual — you implement the logic | Delegates to `benchmark` |
| Faithfulness scoring | ✅ Via `--with-faithfulness` | Manual — you implement the logic | Delegates to `benchmark` |
| CI threshold gating | ✅ `main()` returns `0` on pass | You define exit codes yourself | Possible, with more wiring |
| Custom query files | ✅ Supported | Depends on your script | Depends on configuration |
| Purpose-built public API | ✅ `main()` in `benchmark.py` | ❌ None | Indirect |
| Suitable for exploratory work | ⚠️ Overkill for one-off runs | ✅ Fast to prototype | ❌ Too much overhead |

## When to use `benchmark`

Use `benchmark` when all of the following are true:

- You need repeatable, structured measurement of retrieval quality — precision, recall, or faithfulness — against a defined query set.
- You want CI to fail automatically when results drop below a threshold. `main()` returns `0` on success, making it a natural fit for pipeline exit-code checks.
- You are running faithfulness evaluation alongside retrieval scoring and want both in a single invocation (`--with-faithfulness`).
- You have a custom query file that defines the evaluation set.

## When *not* to use `benchmark`

- **Exploratory or one-off evaluation.** If you are experimenting with a new retrieval approach and do not yet have a stable query file or threshold, a throwaway script avoids the overhead of wiring up `benchmark` for a single run.
- **Multi-feature pipelines.** If your evaluation spans concerns beyond retrieval and faithfulness, use the orchestration layer above `benchmark` rather than calling it directly. Calling `benchmark` in the middle of a broader pipeline couples your orchestration to its internals.
- **Behavior outside the public API.** If you need evaluation logic that `main()` does not expose, do not patch `benchmark` internals. File an issue or propose an extension point instead.

## Recommendation

`benchmark` is the right choice for any team that treats retrieval quality as a CI gate. The combination of built-in precision/recall/faithfulness scoring, configurable thresholds, and a clean `0`/non-zero exit code from `main()` makes it significantly easier to enforce quality standards than building equivalent logic into a custom script. Choose an ad-hoc script only when you are prototyping and not yet ready to commit to a stable query set or threshold.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
