---
type: comparison
name: eval-comparison
feature: eval
depth: comparison
generated_at: 2026-05-20T03:28:38.751351+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Comparison: Faithfulness judging vs prompt benchmarking

## Context

The `eval` module provides two distinct capabilities that are easy to conflate:

1. **Faithfulness judging** — `FaithfulnessJudge` calls Claude (`claude-sonnet-4-6` by default) with a structured tool-use prompt to score whether each factual claim in a RAG answer is directly supported by the retrieved passages.
2. **Prompt-variant benchmarking** — `bench_prompts` runs A/B comparisons across prompt variants against a golden set, letting you measure which prompt formulation produces better answers before shipping.

These two capabilities solve different problems and compose well together: you can run `bench_prompts` to compare variants, then use `FaithfulnessJudge` to score the outputs for grounding.

## Feature comparison

| Capability | Faithfulness judging (`FaithfulnessJudge`) | Prompt benchmarking (`bench_prompts`) |
|---|---|---|
| **Primary question answered** | "Are these answer claims supported by the retrieved passages?" | "Which prompt variant produces better answers?" |
| **Input** | A query, an answer string, and one or more retrieved passages | A golden set of queries and expected outputs |
| **Output** | `FaithfulnessResult` with per-claim `supported_claims`, `unsupported_claims`, a float `score`, and `reasoning` | Aggregate metrics across prompt variants |
| **Strictness** | Strict by design — inference beyond the passages and outside knowledge are marked `UNSUPPORTED` | Depends on the scoring function you pair it with |
| **Extended reasoning** | Optional: `use_thinking=True` activates Claude's extended thinking within a configurable token budget | Not applicable |
| **Async support** | Yes — `FaithfulnessJudge` accepts an `AsyncAnthropic` client | Runs via `main()` CLI entry point |
| **Granularity** | Claim-level: exposes `total_claims`, `supported_claims`, `unsupported_claims` per answer | Answer-level aggregate across the benchmark set |
| **Best suited for** | CI checks on individual RAG responses; hallucination detection in production | Pre-release prompt selection; offline golden-set regression |

## Tradeoffs to understand

**`FaithfulnessJudge` makes one LLM call per scored answer.** If you score a large benchmark set claim-by-claim, costs and latency scale linearly with the number of answers. Batching at the `bench_prompts` layer and scoring selectively with `FaithfulnessJudge` is more economical than scoring every intermediate output.

**Strictness is intentional, not configurable.** The judge system prompt explicitly marks any claim relying on "outside knowledge, reasonable inference beyond what the passages say, or invented details" as `UNSUPPORTED`. You cannot relax this threshold through the public API — if your use case requires fuzzy matching, you need a different scoring approach.

**`use_thinking=True` increases accuracy at higher token cost.** Extended thinking gives the judge more reasoning capacity for complex, multi-hop claims, but it consumes additional tokens against `thinking_budget_tokens`. Reserve it for answers where faithfulness is ambiguous, not as a default.

**`bench_prompts` is the right entry point for comparative work; `FaithfulnessJudge` is the right entry point for per-answer verdicts.** Using `FaithfulnessJudge` as a substitute for a benchmark harness means writing your own aggregation logic from scratch.

## When to use each

**Use `FaithfulnessJudge` when you need to:**
- Detect hallucinations in a specific RAG answer at runtime or in CI.
- Get an auditable, claim-level breakdown (`supported_claims`, `unsupported_claims`, `reasoning`) you can log or surface to users.
- Score answers programmatically from Python using an existing `AsyncAnthropic` client.

**Use `bench_prompts` when you need to:**
- Compare two or more prompt variants against a fixed golden set before a release.
- Run evaluation from the command line (`main()` returns `0` on success) without writing Python integration code.
- Establish a regression baseline you can re-run on future prompt changes.

**Use both together when you need to:**
- Run a prompt A/B benchmark *and* verify that the winning variant's answers are actually grounded — not just fluent or high-scoring on surface metrics.

If your work does not involve RAG answer grounding or prompt-variant selection, neither entry point is the right fit. Consider whether the problem belongs in retrieval, generation, or orchestration instead.

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
