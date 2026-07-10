---
type: concept
name: eval-concept
feature: eval
depth: concept
generated_at: 2026-07-10T13:05:28.714688+00:00
source_hash: 15853b482b28c9a8b23b9156fb0655f28895187b4de4d1e26cdb96202d404731
status: generated
---

# Eval

The eval subsystem measures whether a RAG pipeline's answers are grounded in the passages it retrieved — catching hallucinations before they reach users.

## The faithfulness problem

A RAG system can retrieve relevant passages and still produce an answer that invents details not found in those passages. Faithfulness evaluation catches this by decomposing each answer into atomic factual claims and checking every claim against the retrieved context. A claim is **supported** only if a passage explicitly states it — inference beyond what the passages say, or invented workflow names, CLI flags, or API shapes, all count as **unsupported**.

## Core components

**`FaithfulnessJudge`** is the entry point for scoring. You pass it a query, the answer under review, and one or more retrieved passages. Internally it calls Claude with a strict system prompt and uses tool-use to collect a structured verdict. It exposes a single async method, `score()`, and resolves its model from the `ATTUNE_MODEL_PREMIUM` tier by default.

**`FaithfulnessResult`** is the structured verdict that `score()` returns. It carries:

| Field | Type | What it tells you |
|---|---|---|
| `score` | `float` | Fraction of claims that are supported (0.0–1.0) |
| `supported_claims` | `list[str]` | Claims the passages explicitly back |
| `unsupported_claims` | `list[str]` | Claims that go beyond the retrieved context |
| `reasoning` | `str` | The judge's explanation for each decision |
| `model` | `str` | Which model produced the verdict |
| `thinking_used` | `bool` | Whether extended thinking was enabled |

The `total_claims` property gives you `len(supported_claims) + len(unsupported_claims)` in one call.

## How the pieces fit together

When you call `FaithfulnessJudge.score()`, the judge:

1. Formats the query, passages, and answer into a structured prompt using `_JUDGE_USER_TEMPLATE`.
2. Sends that prompt to Claude under the strict `_JUDGE_SYSTEM_PROMPT`, which instructs the model to decompose the answer into atomic claims and mark each as supported or unsupported.
3. Receives the verdict via the `report_faithfulness` tool call and packages it into a `FaithfulnessResult`.

Setting `use_thinking=True` in `score()` enables Claude's extended thinking for harder judgments, at the cost of additional tokens (controlled by `thinking_budget_tokens`).

## When eval matters

Use the eval subsystem when you need to:

- **Detect hallucinations in a running pipeline.** Score answers at inference time and surface low-`score` results for review.
- **Run A/B benchmarks on prompt variants.** The harness supports golden-set testing so you can compare how different retrieval or prompting strategies affect faithfulness before deploying a change.
- **Set a quality gate in CI.** The `main()` entry point returns exit code `0` on success, making it straightforward to fail a build when faithfulness drops below a threshold.

## Related topics

- **FAQ**: `eval` in tests — common questions about using `FaithfulnessJudge` in test suites
- **Reference**: `FaithfulnessJudge` and `FaithfulnessResult` — complete field list, default values, and method signatures
