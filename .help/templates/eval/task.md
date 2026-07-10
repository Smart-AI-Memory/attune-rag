---
type: task
name: eval-task
feature: eval
depth: task
generated_at: 2026-07-10T13:05:28.720004+00:00
source_hash: 15853b482b28c9a8b23b9156fb0655f28895187b4de4d1e26cdb96202d404731
status: generated
---

# Work with eval

Use the eval module when you need to score RAG answers for faithfulness, run prompt A/B benchmarks against a golden set, or change how the LLM judge model is resolved.

## Prerequisites

- Access to the project source code
- An `AsyncAnthropic` client or a valid `ATTUNE_MODEL_PREMIUM` API key for the judge
- Familiarity with `src/attune_rag/eval/__init__.py` and its sibling modules

## Steps

1. **Identify which eval capability you need.**
   The module has two distinct entry points:
   - **Faithfulness scoring** — `FaithfulnessJudge.score()` in `src/attune_rag/eval/faithfulness.py` calls Claude with tool-use to decompose an answer into atomic claims and label each one as supported or unsupported by the retrieved passages.
   - **Prompt benchmarking** — `main()` in `src/attune_rag/eval/bench_prompts.py` runs prompt variants against a golden set and returns exit code `0` on success.

2. **Read the function signature and return type before editing.**
   For `FaithfulnessJudge.score()`, confirm the expected arguments:

   ```python
   score(
       self,
       query: str,
       answer: str,
       passages: str | list[str],
       max_tokens: int = 3072,
       *,
       use_thinking: bool = False,
       thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS,
   ) -> FaithfulnessResult
   ```

   `FaithfulnessResult` exposes `score`, `supported_claims`, `unsupported_claims`, `reasoning`, `model`, `thinking_used`, and the computed property `total_claims`.

3. **Instantiate `FaithfulnessJudge` with your credentials.**
   Pass an existing `AsyncAnthropic` client, or let the constructor resolve credentials from the environment:

   ```python
   from attune_rag.eval.faithfulness import FaithfulnessJudge

   judge = FaithfulnessJudge()          # uses ATTUNE_MODEL_PREMIUM by default
   # or
   judge = FaithfulnessJudge(api_key="sk-...", model="claude-3-5-sonnet-20241022")
   ```

4. **Call `score()` and inspect the result.**

   ```python
   result = await judge.score(
       query="What is the retention policy?",
       answer="Data is kept for 30 days.",
       passages=["Our system retains data for 30 days after ingestion."],
   )

   print(result.score)               # float between 0.0 and 1.0
   print(result.supported_claims)    # list of claim strings
   print(result.unsupported_claims)  # list of claim strings
   print(result.total_claims)        # int, computed property
   ```

5. **To change the default judge model, update `default_judge_model()`.**
   Open `src/attune_rag/eval/faithfulness.py` and edit `default_judge_model()`. This function resolves the `ATTUNE_MODEL_PREMIUM` environment variable. Update the fallback value or resolution logic there — do not hardcode a model name elsewhere.

6. **To modify the benchmark entry point, edit `main()` in `bench_prompts.py`.**
   Open `src/attune_rag/eval/bench_prompts.py`. `main()` accepts an optional `argv: list[str] | None` and returns an integer exit code. Preserve that signature so the CLI integration continues to work.

7. **Run the eval tests.**

   ```bash
   pytest -k "eval"
   ```

## Key files

| File | Purpose |
|---|---|
| `src/attune_rag/eval/__init__.py` | Public exports: `FaithfulnessJudge`, `FaithfulnessResult` |
| `src/attune_rag/eval/faithfulness.py` | Judge logic, `default_judge_model()`, `FaithfulnessResult` dataclass |
| `src/attune_rag/eval/bench_prompts.py` | Prompt A/B benchmark runner, `main()` entry point |

## Verify success

After running `pytest -k "eval"`, all tests pass with no failures or errors. When you call `judge.score()` directly, `result.score` is a float between `0.0` and `1.0` and `result.total_claims` equals `len(result.supported_claims) + len(result.unsupported_claims)`.
