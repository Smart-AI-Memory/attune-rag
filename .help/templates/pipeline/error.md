---
type: error
name: pipeline-error
feature: pipeline
depth: error
generated_at: 2026-05-20T03:20:40.525187+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: verified
---

# Pipeline errors

## Common error signatures

Failures in the RAG pipeline fall into three broad categories: misconfigured pipeline components, retrieval producing no results, and LLM generation errors. Check exceptions for failed calls and inspect `fallback_used` and `abstained` on successful calls with zero hits.

Watch for these specific conditions:

- **`corpus` is `None` or unset** — `RagPipeline` accepts `corpus=None` in `__init__`, and loads the default corpus on demand. Install the `[attune-help]` extra or pass a corpus explicitly.
- **No grounding context found** — Check `fallback_used` for zero retrieval hits. Strict queries also set `abstained=True` and return a local refusal during generation. Other variants retain the fallback prompt.
- **Invalid `prompt_variant`** — `run()` and `run_and_generate()` accept a `prompt_variant` argument. Passing a value not present in `PROMPT_VARIANTS` raises `ValueError` before retrieval.
- **`provider` resolution failure in `run_and_generate()`** — The `provider` argument accepts either an `LLMProvider` instance or a string name. An unrecognised name raises an error when generation needs a provider. Strict zero-hit queries bypass provider initialization.
- **`k` too large** — If `k` exceeds the number of documents in the corpus, retrieval may return fewer results than expected. This silently reduces context rather than raising an exception, which can lower `confidence` in the result.

## Where errors originate

Trace the failure to the specific method before walking the call stack further.

- **`RagPipeline.run(query, k, prompt_variant)`** — Orchestrates corpus retrieval, optional query expansion and reranking, and prompt assembly. Errors here are most often caused by a missing or misconfigured `corpus`, `retriever`, or `expander`.
- **`RagPipeline.run_and_generate(query, provider, ...)`** — Extends `run()` with LLM generation. Errors here can originate in either the retrieval/prompt layer or the LLM provider call. Check `RagResult` fields from the returned tuple to distinguish the two.
- **`RagPipeline.corpus` property** — Accessing this property loads the default help corpus if you omitted `corpus`.

## How to diagnose

1. **Inspect `RagResult` fields first.** Before examining the traceback, check `fallback_used`, `confidence`, and `context` on the returned `RagResult`. A `fallback_used=True` result with an empty `context` confirms retrieval found nothing — the problem is in the corpus or query, not the pipeline code itself.

2. **Check that `corpus` and `retriever` are not `None`.** Both have defaults. Print `pipeline.corpus` before calling `run()` to confirm the pipeline is fully initialised.

3. **Validate `prompt_variant` against `PROMPT_VARIANTS`.** If the traceback points to prompt assembly, confirm the variant string you're passing is a key in `PROMPT_VARIANTS`. Log or print the value immediately before the `run()` call.

4. **Trace `run_and_generate()` failures to their layer.** This method returns `tuple[str, RagResult]`. If the coroutine raises an exception, inspect its traceback to distinguish retrieval errors from answer-provider errors.

5. **Check `claim_citations` and `used_native_citations` for generation-side issues.** When `use_native_citations=True` is passed to `run_and_generate()`, the pipeline delegates citation extraction to the LLM. If `used_native_citations` is `False` on the result despite being requested, check for zero retrieval hits and confirm the provider supports native citations.

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
