---
type: warning
name: eval-warning
feature: eval
depth: warning
generated_at: 2026-05-20T03:28:38.737092+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8472542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Eval cautions

## What to watch for

`FaithfulnessJudge` calls the Claude API using tool-use to decompose answers into atomic claims and classify each one as supported or unsupported against retrieved passages. The scoring logic, judge model, and prompt strictness all affect results in ways that are easy to overlook when integrating or extending this module.

## Risk areas

### Strict-by-design scoring underestimates faithful answers that infer beyond passages

The judge system prompt instructs the model to mark a claim as `UNSUPPORTED` whenever it relies on inference, outside knowledge, or any detail not explicitly stated in the retrieved passages. A correct, reasonable answer that synthesizes across passages will lose score for claims the model considers implicit rather than explicit. This is intentional strictness — but it means a low `FaithfulnessResult.score` does not always indicate a hallucination; it may indicate an answer that is too generative for the judge's threshold.

**Mitigation:** When interpreting results, review `unsupported_claims` alongside `reasoning` before treating a low score as a quality failure. Use `total_claims` (the `FaithfulnessResult.total_claims` property) to understand score denominator — a result with two unsupported claims out of three is very different from two out of twenty.

### `use_thinking=False` by default may miss multi-hop reasoning errors

`FaithfulnessJudge.score()` sets `use_thinking=False` by default. For answers that synthesize across multiple passages or involve chained inferences, the judge may produce a less reliable verdict without extended thinking enabled. The `thinking_budget_tokens` parameter defaults to `DEFAULT_THINKING_BUDGET_TOKENS` and has no effect unless you also pass `use_thinking=True`.

**Mitigation:** For complex RAG pipelines or benchmark evaluation where accuracy matters most, call `score()` with `use_thinking=True` and set `thinking_budget_tokens` explicitly rather than relying on the default.

### The judge model is pinned to `claude-sonnet-4-6` and affects reproducibility

`DEFAULT_JUDGE_MODEL = 'claude-sonnet-4-6'` is baked into the default for `FaithfulnessJudge.__init__()`. If Anthropic updates the model behind that name, scores for the same query-answer-passage triple can shift between runs without any change to your code.

**Mitigation:** Pass the `model` parameter explicitly when constructing `FaithfulnessJudge` and record the resolved model name from `FaithfulnessJudge.model` in your evaluation logs. `FaithfulnessResult.model` captures the model used per result, so include it when persisting results with `to_dict()`.

### Passing a single string to `passages` skips per-passage structure

`FaithfulnessJudge.score()` accepts `passages` as either `str` or `list[str]`. Passing a single concatenated string instead of a list works, but the judge treats the entire blob as one passage. If a claim appears in a section of that string that is semantically distant from the answer, the judge may still mark it as supported — or, conversely, miss support that a structured list would surface clearly.

**Mitigation:** Always pass `passages` as a `list[str]` with one entry per retrieved chunk. This matches how the judge prompt is designed and produces more consistent claim-level verdicts.

### `max_tokens=2048` can truncate judge output on answers with many claims

The default `max_tokens=2048` for `FaithfulnessJudge.score()` limits the tool-use response. Answers that decompose into a large number of atomic claims may cause the model to truncate its `report_faithfulness` tool call, resulting in an incomplete `FaithfulnessResult` where only a subset of claims are classified.

**Mitigation:** For long answers or high-claim-count benchmarks, increase `max_tokens` proportionally. A rough heuristic is 150–200 tokens per expected atomic claim.

## How to avoid problems

1. **Log `FaithfulnessResult.model` with every result.** Because the judge model can be overridden at construction time and is also captured per-result, always serialize results via `to_dict()` and retain the `model` field. This makes score comparisons across runs meaningful.

2. **Separate benchmark runs from production evaluation.** The `bench_prompts` module is designed for golden-set A/B testing of prompt variants, not for per-request production scoring. Running benchmark workloads against a shared `FaithfulnessJudge` instance can interfere with latency budgets and API rate limits in production paths.

3. **Treat private symbols as unstable.** Names beginning with `_` — including `_JUDGE_SYSTEM_PROMPT` and `_JUDGE_USER_TEMPLATE` — are not part of the public API (`__all__` exports only `FaithfulnessJudge` and `FaithfulnessResult`). If you customize the judge prompt by referencing these constants directly, your integration will break silently when the prompt is revised.

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
