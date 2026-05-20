---
type: faq
name: eval-faq
feature: eval
depth: faq
generated_at: 2026-05-20T03:28:38.742095+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Eval FAQ

## What does the eval feature do?

It provides two related capabilities: a **faithfulness judge** that scores whether each claim in a RAG answer is supported by the retrieved passages, and a **prompt A/B benchmark** for comparing prompt variants against a golden set.

## What is faithfulness scoring?

`FaithfulnessJudge` sends your query, answer, and retrieved passages to Claude (default model: `claude-sonnet-4-6`). Claude decomposes the answer into atomic factual claims and labels each one as supported or unsupported based strictly on the passages — not on outside knowledge or inference. The result is a `FaithfulnessResult` with a `score`, a list of `supported_claims`, a list of `unsupported_claims`, and a `reasoning` string.

## When should I use the faithfulness judge?

Use it whenever you need to detect hallucinations in a RAG pipeline — for example, during CI on a golden set, after changing a retrieval strategy, or when auditing answers in production.

## How strict is the judge?

Very strict. A claim is only marked supported if a retrieved passage **explicitly states it**. Reasonable inference, background knowledge, and details not present in the passages all count as unsupported. When the judge is uncertain, it marks the claim unsupported.

## What does the faithfulness score represent?

`FaithfulnessResult.score` is a float between 0 and 1. It reflects the proportion of atomic claims in the answer that are directly supported by the retrieved passages. You can also inspect `total_claims`, `supported_claims`, and `unsupported_claims` individually.

## How do I run the judge in my code?

```python
from attune_rag.eval.faithfulness import FaithfulnessJudge

judge = FaithfulnessJudge()  # uses ANTHROPIC_API_KEY from env
result = await judge.score(
    query="What is the refund policy?",
    answer="Refunds are processed within 5 business days.",
    passages=["Our refund policy allows returns within 30 days..."],
)
print(result.score, result.unsupported_claims)
```

Pass `use_thinking=True` to `score()` if you want extended reasoning; the `FaithfulnessResult.thinking_used` field will reflect whether it was active.

## What does `to_dict()` return?

`FaithfulnessResult.to_dict()` serializes all fields — `score`, `supported_claims`, `unsupported_claims`, `reasoning`, `model`, `thinking_used`, and `total_claims` — into a plain dictionary, which is useful for logging or writing results to JSON.

## How do I run the eval benchmark from the command line?

Call `main()` (exit code `0` on success) or run the benchmark entry point directly. Check `src/attune_rag/eval/bench_prompts.py` for flags and expected arguments.

## How do I debug a failing evaluation?

Run the eval tests first:

```
pytest -k "eval" -v
```

If the tests pass but your code still misbehaves, add a `logger.debug` statement just before the `judge.score()` call and re-run with logging enabled. Common things to check:

- Are your `passages` non-empty and actually relevant to the query?
- Is the answer a refusal string (for example, "the context does not cover this")? The judge treats refusals as zero claims, so `score` will be `0` and both claim lists will be empty — this is expected behavior, not a bug.
- Is the `model` property on your `FaithfulnessJudge` instance set to what you expect?

## Where are the source files?

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
