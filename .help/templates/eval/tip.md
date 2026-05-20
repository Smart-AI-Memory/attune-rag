---
type: tip
name: eval-tip
feature: eval
depth: tip
generated_at: 2026-05-20T03:28:38.746895+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Tip: working effectively with eval

## Recommendation

Use `FaithfulnessResult.unsupported_claims` to drive prompt iteration, not just the aggregate `score`.

The `score` field (a float between 0 and 1) tells you *how much* of an answer is grounded, but `unsupported_claims` tells you *what* your RAG pipeline is hallucinating. Feeding those specific strings back into your prompt or retrieval logic is faster than tuning blindly against the score alone.

**Why it sticks:** the judge prompt is deliberately strict — a claim is only `SUPPORTED` if a passage *explicitly* states it, not merely implies it — so even a score of 0.8 can mask a pattern of the model inferring details that aren't in your passages.

## How to use it

```python
judge = FaithfulnessJudge()  # defaults to claude-sonnet-4-6
result = judge.score(query, answer, passages)

if result.unsupported_claims:
    print(f"{len(result.unsupported_claims)} / {result.total_claims} claims not grounded:")
    for claim in result.unsupported_claims:
        print(" -", claim)
```

`result.reasoning` gives you the judge's chain-of-thought explanation for each verdict, which is useful for understanding borderline calls.

## Tradeoff

Reading `unsupported_claims` one by one is more work than threshold-gating on `score`. It pays off when you are diagnosing a specific failure mode; it is overkill for a bulk pass/fail CI check where the score alone is sufficient.

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
