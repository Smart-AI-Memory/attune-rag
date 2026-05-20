---
type: note
name: eval-note
feature: eval
depth: note
generated_at: 2026-05-20T03:28:38.748793+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Note: eval

## Context

The `eval` module provides two complementary capabilities: LLM-as-judge faithfulness scoring for RAG answers, and a prompt-variant A/B benchmark harness for golden-set testing.

## How faithfulness scoring works

`FaithfulnessJudge` (in `faithfulness.py`) sends a RAG answer to `claude-sonnet-4-6` using Claude's tool-use API. The judge system prompt instructs the model to decompose the answer into atomic factual claims — one claim per verifiable assertion — and classify each claim as either supported or unsupported by the retrieved passages.

The scoring rules are strict by design:

- A claim is **supported** only if a retrieved passage explicitly states it.
- A claim is **unsupported** if it relies on outside knowledge, reasonable inference, or details not present in the passages (such as workflow names, CLI flags, or API shapes).
- If the answer is a refusal (for example, "the context does not cover this"), the judge treats it as zero claims rather than a supported or unsupported verdict.

Results are returned as a `FaithfulnessResult` dataclass, which records the `score` (a float), the `supported_claims` and `unsupported_claims` lists, `reasoning` text, the `model` used, and whether extended thinking was active (`thinking_used`). The `total_claims` property returns the sum of both claim lists.

Extended thinking is opt-in: pass `use_thinking=True` to `FaithfulnessJudge.score()`. When disabled, `thinking_used` is `False` on every result.

## Benchmark harness

`bench_prompts.py` exposes a `main()` entry point for running prompt-variant A/B comparisons against a golden set. It is separate from the faithfulness judge and does not depend on `FaithfulnessJudge` or `FaithfulnessResult` at runtime.

## Public API boundary

Only `FaithfulnessJudge` and `FaithfulnessResult` are exported from the package (`__all__`). The `main()` function in `bench_prompts.py` is a CLI entry point, not part of the importable surface.

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
