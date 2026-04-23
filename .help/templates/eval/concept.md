---
type: concept
feature: eval
depth: concept
generated_at: 2026-04-23T03:35:59.671106+00:00
source_hash: c51ee93d206242f5b5e8a7ad4ed5772aaf45bc2940d459444097d7a899d4513c
status: generated
---

# Eval

Eval is the evaluation harness that measures RAG answer quality through AI-powered faithfulness scoring and prompt variant testing.

## Core responsibilities

**Faithfulness scoring**: The `FaithfulnessJudge` uses Claude to decompose RAG answers into atomic factual claims and verify each claim against the retrieved passages. Claims are marked as supported (explicitly stated in passages) or unsupported (based on outside knowledge, inference, or invention).

**Prompt benchmarking**: The system tests different prompt variants against golden datasets to identify which approaches produce the most faithful answers.

## Components

**`FaithfulnessJudge`**: The primary scoring engine that takes a user query, generated answer, and retrieved passages, then returns a `FaithfulnessResult` with a score (0.0-1.0), lists of supported and unsupported claims, and reasoning.

**`FaithfulnessResult`**: A structured verdict containing the faithfulness score, claim breakdowns, model reasoning, and metadata. It provides a `total_claims` property and can serialize to a dictionary for storage or reporting.

## Evaluation philosophy

The judge applies strict standards — claims are unsupported unless passages explicitly state them. This catches hallucinations where the model invents workflow names, CLI flags, or API details not present in the context. Refusals ("the context doesn't cover this") are treated as zero claims rather than scoring events.

The system uses Claude's tool-calling capability to ensure structured output and consistent scoring criteria across evaluations.
