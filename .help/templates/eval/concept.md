---
type: concept
name: eval-concept
feature: eval
depth: concept
generated_at: 2026-05-15T20:03:10.201475+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Eval

The `eval` feature gives you a structured way to measure whether a RAG system's answers are actually grounded in the passages it retrieved — and to benchmark prompt variants against a golden set.

## The faithfulness problem

A RAG system can produce fluent, confident-sounding answers that introduce facts not present in any retrieved passage. Without a systematic check, these hallucinations are invisible during development. The `eval` feature surfaces them by decomposing each answer into atomic factual claims and verifying each claim against the source passages independently.

The judge is strict by design: a claim is supported only if a passage **explicitly states it**. Reasonable inference, outside knowledge, and invented details (workflow names, CLI flags, API shapes not in the passages) all count as unsupported.

## Core components

**`FaithfulnessJudge`** drives the evaluation. It calls Claude (`claude-sonnet-4-6` by default) using tool use, passing the original query, the retrieved passages, and the answer under review. Claude decomposes the answer into atomic claims and reports which are supported and which are not.

**`FaithfulnessResult`** holds the verdict for a single answer. Its fields map directly to what you need for analysis:

| Field | Type | What it tells you |
|---|---|---|
| `score` | `float` | Fraction of claims that are supported |
| `supported_claims` | `list[str]` | Claims grounded in the passages |
| `unsupported_claims` | `list[str]` | Claims that go beyond the passages |
| `reasoning` | `str` | The judge's explanation |
| `model` | `str` | Which Claude model produced the verdict |
| `thinking_used` | `bool` | Whether extended thinking was enabled |

`total_claims` is a computed property — the sum of supported and unsupported claims — so you can quickly see the denominator behind `score`.

## How the pieces fit together

When you call `FaithfulnessJudge.score(query, answer, passages)`, the flow is:

1. The judge formats the query, passages, and answer into a structured prompt using `_JUDGE_USER_TEMPLATE`.
2. Claude receives a strict system prompt (`_JUDGE_SYSTEM_PROMPT`) that instructs it to treat refusals as zero claims rather than as supported or unsupported ones.
3. Claude calls the `report_faithfulness` tool with the decomposed claims.
4. The judge returns a `FaithfulnessResult` you can inspect, serialize with `to_dict()`, or aggregate across a golden set.

For cases where answer complexity warrants deeper analysis, pass `use_thinking=True` to enable extended thinking; the `thinking_budget_tokens` parameter controls how much reasoning Claude can do before reporting.

## When this matters

Use `eval` when you need to:

- **Catch hallucinations before they reach users.** Run `FaithfulnessJudge` as part of your CI pipeline against a golden set of query/passage/answer triples.
- **Compare prompt variants.** The A/B benchmark module lets you score two or more prompt variants against the same golden set and compare their faithfulness distributions.
- **Audit a deployed system.** Log `FaithfulnessResult` records in production to track score trends over time as your retrieval corpus or prompt changes.

## Related topics

- **FAQ**: `eval` in tests — how to use `FaithfulnessJudge` in unit and integration tests
- **Reference**: `eval` — all fields, parameters, and default constants
- **Quickstart**: `eval` — score your first answer in five steps
