---
type: task
name: eval-task
feature: eval
depth: task
generated_at: 2026-05-20T03:28:38.722384+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Score RAG answer faithfulness with eval

Use the eval harness when you want to measure how well your RAG pipeline grounds its answers in retrieved passages, or to benchmark prompt variants against a golden set.

## Prerequisites

- An Anthropic API key, or an `AsyncAnthropic` client instance
- Python dependencies installed (the `eval` module must be importable)
- Retrieved passages and the answers you want to judge

## Score an answer for faithfulness

1. **Import `FaithfulnessJudge` and instantiate it.**

   ```python
   from attune_rag.eval import FaithfulnessJudge

   judge = FaithfulnessJudge(api_key="YOUR_API_KEY")
   ```

   By default the judge uses the `claude-sonnet-4-6` model. Pass `model=` to override it.

2. **Call `score()` with your query, answer, and passages.**

   ```python
   result = await judge.score(
       query="What is the retention policy for audit logs?",
       answer="Audit logs are retained for 90 days.",
       passages=["Audit logs are kept for a period of 90 days before deletion."],
   )
   ```

   Pass `passages` as a single string or a list of strings. Set `use_thinking=True` to enable extended reasoning if you need higher-confidence verdicts on ambiguous claims.

3. **Inspect the `FaithfulnessResult`.**

   ```python
   print(result.score)               # float between 0.0 and 1.0
   print(result.supported_claims)    # list of claims the passages back up
   print(result.unsupported_claims)  # list of claims not grounded in passages
   print(result.reasoning)           # judge's chain-of-thought explanation
   print(result.total_claims)        # total_claims = len(supported) + len(unsupported)
   ```

   Convert the result to a plain dictionary with `result.to_dict()` for logging or serialization.

4. **Run the prompt-variant benchmark** (optional).

   Execute the benchmark entry point to compare prompt variants against your golden set:

   ```bash
   pytest -k "eval"
   ```

   Alternatively, call `main()` from `bench_prompts.py` directly in your test harness. A return value of `0` indicates a successful run.

## Verify success

The task succeeded when:

- `result.score` is a float and `result.total_claims` equals `len(result.supported_claims) + len(result.unsupported_claims)`.
- `result.unsupported_claims` is empty (or within your acceptable threshold) for answers you expect to be fully grounded.
- `main()` returns `0` when you run the benchmark.

## Key files

| File | Purpose |
|---|---|
| `src/attune_rag/eval/__init__.py` | Public exports: `FaithfulnessJudge`, `FaithfulnessResult` |
| `src/attune_rag/eval/faithfulness.py` | Judge logic, system prompt, and `FaithfulnessResult` dataclass |
| `src/attune_rag/eval/bench_prompts.py` | Prompt A/B benchmark harness and `main()` entry point |
