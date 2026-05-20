---
type: troubleshooting
name: eval-troubleshooting
feature: eval
depth: troubleshooting
generated_at: 2026-05-20T03:28:38.739602+00:00
source_hash: 9d4d7626c287ef8da26b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Troubleshoot eval

## Before you start

The `eval` module provides two capabilities: LLM-as-judge faithfulness scoring (via `FaithfulnessJudge`) and prompt-variant A/B benchmarking. `FaithfulnessJudge` calls the Claude API using tool use to decompose an answer into atomic claims and classify each as supported or unsupported against your retrieved passages. Most failures trace back to one of three root causes: an invalid or missing API key, malformed inputs to `FaithfulnessJudge.score()`, or an unexpected response from the judge model.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `anthropic.AuthenticationError` or `401` | Whether `ANTHROPIC_API_KEY` is set in your environment, or whether you passed a valid `api_key` argument to `FaithfulnessJudge()` |
| `FaithfulnessResult.score` is `0.0` and `unsupported_claims` contains everything | Whether your `passages` argument is non-empty and actually contains the source text; an empty string passes validation but gives the judge nothing to ground claims against |
| `FaithfulnessResult.total_claims` is `0` | Whether the `answer` argument is a refusal or contains no factual assertions — the judge prompt treats refusals as zero claims by design |
| `asyncio` errors or `RuntimeError: no running event loop` | Whether you are calling `FaithfulnessJudge.score()` outside an async context; the underlying client is `AsyncAnthropic` |
| Timeout / `httpx.ReadTimeout` | Whether `timeout` (default `DEFAULT_JUDGE_TIMEOUT_SECONDS`) is long enough for your model and token budget; extended thinking increases latency significantly |
| `use_thinking=True` produces unexpected results | Whether `thinking_budget_tokens` is sufficient; too small a budget causes the model to truncate its reasoning before calling the `report_faithfulness` tool |
| `main()` returns a non-zero exit code | The stderr output — `main()` is documented to return `0` on success, so any other value indicates an unhandled exception or argument parsing failure |

## Step-by-step diagnosis

1. **Reproduce the failure with a minimal call.**
   Construct the smallest possible invocation of `FaithfulnessJudge.score()` directly — a single short query, a single passage, and a single-sentence answer. If the failure disappears, the problem is in your upstream data, not the judge itself.

2. **Check the `FaithfulnessResult` fields before assuming a bug.**
   Print or call `.to_dict()` on the result and inspect all fields: `score`, `supported_claims`, `unsupported_claims`, `reasoning`, `model`, and `thinking_used`. The `reasoning` field contains the judge's chain-of-thought and often explains unexpected scores directly.

3. **Verify your API key and model.**
   Confirm the judge is using the model you expect:
   ```python
   judge = FaithfulnessJudge()
   print(judge.model)  # should print 'claude-sonnet-4-6' unless overridden
   ```
   Then verify the key is reachable:
   ```bash
   echo $ANTHROPIC_API_KEY   # must be non-empty
   python -c "import anthropic; anthropic.Anthropic().models.list()"
   ```

4. **Enable debug logging.**
   The `anthropic` SDK and `httpx` both respect Python's `logging` module. Set the level to `DEBUG` before instantiating the judge to see the raw request and response:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
   Look for the `report_faithfulness` tool call and its arguments in the output — a missing tool call means the model did not follow the judge prompt.

5. **Run the existing test suite.**
   ```bash
   pytest -k "eval" -v
   ```
   If tests that exercise `FaithfulnessJudge` or `bench_prompts` fail, the output will point to the specific assertion. Use the test fixtures as a baseline for your own reproduction case.

## Common fixes

- **Empty or whitespace-only passages.** Pass a non-empty string or a list with at least one non-empty element. An empty `passages` argument causes the judge to mark every claim unsupported because there is nothing to verify against.
  ```python
  # Bad
  result = await judge.score(query, answer, passages="")
  # Good
  result = await judge.score(query, answer, passages=retrieved_text)
  ```

- **Calling `score()` synchronously.** `FaithfulnessJudge` uses `AsyncAnthropic` internally. Wrap the call in an async runner if you are outside an event loop:
  ```python
  import asyncio
  result = asyncio.run(judge.score(query, answer, passages))
  ```

- **Timeout on large inputs or extended thinking.** Increase the `timeout` parameter when instantiating the judge:
  ```python
  judge = FaithfulnessJudge(timeout=120.0)
  ```
  If you use `use_thinking=True`, also increase `thinking_budget_tokens` above its default to give the model room to complete its reasoning before calling the tool.

- **Wrong model loaded.** If a wrapper or environment variable overrides the default, pass the model explicitly:
  ```python
  judge = FaithfulnessJudge(model="claude-sonnet-4-6")
  ```

- **Dependency version mismatch.** A breaking change in the `anthropic` SDK can alter tool-use behavior. Check the installed version:
  ```bash
  pip show anthropic
  ```
  Pin to the version your project was tested against if you see unexpected response shapes.

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
