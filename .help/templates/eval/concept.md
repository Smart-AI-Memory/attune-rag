---
type: concept
name: eval-concept
feature: eval
depth: concept
generated_at: 2026-05-20T03:28:38.716478+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Eval

## Overview

The `eval` feature is an evaluation harness that measures whether a RAG system's answers are grounded in the passages it retrieved — and lets you benchmark competing prompt variants against a golden set.

At its core, the workflow is: you pass a query, an answer, and the retrieved passages to `FaithfulnessJudge`; the judge calls Claude (`claude-sonnet-4-6` by default) using tool-use to decompose the answer into atomic factual claims; each claim is marked **supported** or **unsupported** based solely on what the passages explicitly state; and the results come back as a `FaithfulnessResult` you can inspect or serialize.

## Faithfulness judging

`FaithfulnessJudge` scores a RAG answer by asking Claude to act as a strict faithfulness judge. The system prompt instructs the model to:

- Break the answer into one claim per verifiable assertion, ignoring stylistic framing and pleasantries.
- Mark a claim **supported** only when a retrieved passage explicitly states it — inference beyond the text and outside knowledge both count as unsupported.
- Treat a refusal answer (for example, "the context does not cover this") as zero claims rather than a supported or unsupported verdict.

You can optionally enable extended thinking via `use_thinking=True` on `score()`, which sets a token budget for the model's reasoning before it calls the tool.

## The faithfulness verdict

`FaithfulnessResult` is the dataclass returned by every `score()` call. Its fields give you a complete picture of one answer:

| Field | Type | What it tells you |
|---|---|---|
| `score` | `float` | Fraction of claims that are supported (supported ÷ total) |
| `supported_claims` | `list[str]` | Claims the passages directly back up |
| `unsupported_claims` | `list[str]` | Claims that rely on outside knowledge or inference |
| `reasoning` | `str` | The judge's explanation of its verdicts |
| `model` | `str` | Which Claude model produced the verdict |
| `thinking_used` | `bool` | Whether extended thinking was active |

The `total_claims` property returns `len(supported_claims) + len(unsupported_claims)`, giving you a quick way to gauge answer complexity alongside the score.

## Prompt A/B benchmarking

In addition to per-answer judging, `eval` includes a prompt-variant benchmark that runs two or more prompt formulations against the same golden set and compares their faithfulness scores. This lets you make data-driven decisions about prompt changes rather than relying on manual spot-checks.

## When this matters

Use `eval` when you need to:

- **Detect hallucinations** — a low `score` or a non-empty `unsupported_claims` list signals that your retriever or generator is introducing facts not present in the retrieved context.
- **Regression-test prompt changes** — run the A/B benchmark before and after a prompt edit to confirm faithfulness does not degrade.
- **Audit a deployed RAG pipeline** — serialize results with `FaithfulnessResult.to_dict()` and store them alongside your answer logs for offline analysis.
