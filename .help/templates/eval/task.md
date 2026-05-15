---
type: task
name: eval-task
feature: eval
depth: task
generated_at: 2026-05-15T20:03:10.205581+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Work with eval

Use the eval module when you need to score RAG answers for faithfulness or run prompt A/B benchmarks against a golden set.

## Prerequisites

- Access to the project source code
- An Anthropic API key (passed to `FaithfulnessJudge` via `api_key` or the environment)
- Familiarity with the files under `src/attune_rag/eval/`

## Score a RAG answer for faithfulness

1. **Instantiate `FaithfulnessJudge`.**
   Import the class from `src/attune_rag/eval/faithfulness.py` and create an instance, passing your `AsyncAnthropic` client or API key and the model you want to use (default: `claude-sonnet-4-6`):

   ```python
   from attune_rag.eval.faithfulness import FaithfulnessJudge

   judge = FaithfulnessJudge(api_key="YOUR_KEY")
   ```

2. **Call `score()` with the query, answer, and retrieved passages.**
   Pass the user query, the answer under review, and one or more retrieved passages. Set `use_thinking=True` if you want extended reasoning:

   ```python
   result = await judge.score(
       query="What is the refund policy?",
       answer="Refunds are issued within 30 days.",
       passages=["Our refund policy allows returns within 30 days of purchase."],
   )
   ```

3. **Inspect the `FaithfulnessResult`.**
   The returned dataclass exposes the fields you need to evaluate answer quality:

   | Field | Type | What it tells you |
   |---|---|---|
   | `score` | `float` | Fraction of claims that are supported |
   | `supported_claims` | `list[str]` | Claims grounded in the passages |
   | `unsupported_claims` | `list[str]` | Claims that go beyond the passages |
   | `reasoning` | `str` | Judge's explanation |
   | `total_claims` | `int` (property) | Total atomic claims evaluated |
   | `thinking_used` | `bool` | Whether extended thinking ran |

## Run the prompt benchmark

1. **Invoke `main()` from `bench_prompts.py`.**
   Call it directly in Python or run the module from the command line. `main()` returns `0` on success:

   ```bash
   python -m attune_rag.eval.bench_prompts
   ```

2. **Modify `main()` to add or adjust prompt variants.**
   Open `src/attune_rag/eval/bench_prompts.py` and edit the prompt variants or golden-set inputs inside `main()`. Keep the return type as `int` and return `0` on a clean run.

## Run the tests

After any change, run the eval test suite to catch regressions before they reach other developers:

```bash
pytest -k "eval"
```

A passing run with no errors confirms your changes are correct.

## Key files

| File | Purpose |
|---|---|
| `src/attune_rag/eval/__init__.py` | Public exports: `FaithfulnessJudge`, `FaithfulnessResult` |
| `src/attune_rag/eval/faithfulness.py` | `FaithfulnessJudge` and `FaithfulnessResult` implementation |
| `src/attune_rag/eval/bench_prompts.py` | Prompt A/B benchmark entry point (`main()`) |
