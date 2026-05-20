---
type: error
name: eval-error
feature: eval
depth: error
generated_at: 2026-05-20T03:28:38.731120+00:00
source_hash: 9d4d7626c287ef8da46b5caa6cd8470542d6e7b12acbf7d3e678d0c442cc9f43
status: generated
---

# Eval errors

## Common error signatures

Errors in the eval module fall into three categories: API failures when `FaithfulnessJudge` calls Claude, malformed responses that prevent JSON parsing into `FaithfulnessResult`, and invalid inputs to `FaithfulnessJudge.score()`.

Concrete signatures to watch for:

- **`anthropic.APIConnectionError` / `anthropic.APITimeoutError`** — The judge call exceeded `DEFAULT_JUDGE_TIMEOUT_SECONDS` or the network was unreachable. Check your API key, network connectivity, and whether you passed a custom `timeout` to `FaithfulnessJudge.__init__()`.
- **`anthropic.AuthenticationError`** — No valid API key was found. Pass `api_key=` explicitly to `FaithfulnessJudge()` or set the `ANTHROPIC_API_KEY` environment variable.
- **`ValueError` on `passages`** — `FaithfulnessJudge.score()` accepts `passages` as either a `str` or `list[str]`. Passing `None` or an incompatible type causes a failure before the API call is made.
- **Malformed tool-use response** — If Claude does not call `report_faithfulness`, parsing into `FaithfulnessResult` fails. This can occur when `max_tokens` is too low to complete the structured output or when `thinking_budget_tokens` is misconfigured alongside `use_thinking=True`.
- **`main()` non-zero exit** — `main()` returns `0` on success. Any other exit code indicates that the benchmark run in `bench_prompts.py` did not complete cleanly; inspect stderr for the underlying exception.

## How to diagnose

1. **Identify whether the failure is in the judge call or in result parsing.** A traceback rooted in `faithfulness.py` inside `score()` points to the API call or prompt construction. A traceback during `FaithfulnessResult` field access (e.g., `.score`, `.supported_claims`, `.total_claims`) points to a parsing or deserialization problem.

2. **Check `FaithfulnessResult` fields for sentinel values.** If scoring completes but results look wrong, inspect `result.supported_claims` and `result.unsupported_claims` directly. An empty `supported_claims` list with a low `score` means the judge found no passage-backed claims — this is expected behavior when the answer contains hallucinated details (workflow names, CLI flags, or API shapes not present in the retrieved passages), not a bug.

3. **Verify `use_thinking` token budgets.** When you call `score(..., use_thinking=True)`, the response must fit within `max_tokens`. If `thinking_budget_tokens` approaches or exceeds `max_tokens`, the model may not produce a valid `report_faithfulness` tool call. Increase `max_tokens` (default is `2048`) or reduce `thinking_budget_tokens`.

4. **Confirm the model name.** `FaithfulnessJudge` defaults to `DEFAULT_JUDGE_MODEL` (`claude-sonnet-4-6`). If you pass a custom `model=` string that the Anthropic API does not recognise, the call fails with a `404` or `invalid_request_error`. Check `judge.model` to confirm which model is active.

5. **Inspect the passages you passed.** The judge prompt inserts `passages` verbatim into `_JUDGE_USER_TEMPLATE`. Empty or very short passages cause the model to mark every claim as `UNSUPPORTED` by design — the system prompt instructs it to be strict. Verify that retrieval returned meaningful content before calling `score()`.

## Source files

- `src/attune_rag/eval/__init__.py`
- `src/attune_rag/eval/faithfulness.py`
- `src/attune_rag/eval/bench_prompts.py`

**Tags:** `eval`, `faithfulness`, `judge`, `hallucination`, `scoring`
