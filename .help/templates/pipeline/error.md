---
type: error
name: pipeline-error
feature: pipeline
depth: error
generated_at: 2026-05-20T03:20:40.525187+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline errors

## Common error signatures

Failures in the RAG pipeline fall into three broad categories: misconfigured pipeline components, retrieval producing no results, and LLM generation errors. They typically appear as exceptions raised from `RagPipeline.run()` or `RagPipeline.run_and_generate()` and are visible in the returned `RagResult` fields — particularly `fallback_used: bool` and `confidence: float`.

Watch for these specific conditions:

- **`corpus` is `None` or unset** — `RagPipeline` accepts `corpus=None` in `__init__`, but calling `run()` without a valid `CorpusProtocol` will fail when the pipeline attempts retrieval. Check `pipeline.corpus` before calling `run()`.
- **No grounding context found** — When retrieval returns no hits, the pipeline falls back to `FALLBACK_PROMPT_TEMPLATE`. The returned `RagResult` will have `fallback_used=True`. This is expected behavior, not an exception, but it signals that the query did not match corpus content.
- **Invalid `prompt_variant`** — `run()` and `run_and_generate()` accept a `prompt_variant` argument. Passing a value not present in `PROMPT_VARIANTS` will raise a `KeyError` or `ValueError` during prompt assembly.
- **`provider` resolution failure in `run_and_generate()`** — The `provider` argument accepts either an `LLMProvider` instance or a string name. An unrecognised string will raise an error before any LLM call is made.
- **`k` too large** — If `k` exceeds the number of documents in the corpus, retrieval may return fewer results than expected. This silently reduces context rather than raising an exception, which can lower `confidence` in the result.

## Where errors originate

Trace the failure to the specific method before walking the call stack further.

- **`RagPipeline.run(query, k, prompt_variant)`** — Orchestrates corpus retrieval, optional query expansion and reranking, and prompt assembly. Errors here are most often caused by a missing or misconfigured `corpus`, `retriever`, or `expander`.
- **`RagPipeline.run_and_generate(query, provider, ...)`** — Extends `run()` with LLM generation. Errors here can originate in either the retrieval/prompt layer or the LLM provider call. Check `RagResult` fields from the returned tuple to distinguish the two.
- **`RagPipeline.corpus` property** — Accessing this property when `corpus` was not supplied to `__init__` will raise an `AttributeError` or similar. Validate the corpus is set before use.

## How to diagnose

1. **Inspect `RagResult` fields first.** Before examining the traceback, check `fallback_used`, `confidence`, and `context` on the returned `RagResult`. A `fallback_used=True` result with an empty `context` confirms retrieval found nothing — the problem is in the corpus or query, not the pipeline code itself.

2. **Check that `corpus` and `retriever` are not `None`.** Both are optional constructor parameters, but `run()` requires them at call time. Print `pipeline.corpus` before calling `run()` to confirm the pipeline is fully initialised.

3. **Validate `prompt_variant` against `PROMPT_VARIANTS`.** If the traceback points to prompt assembly, confirm the variant string you're passing is a key in `PROMPT_VARIANTS`. Log or print the value immediately before the `run()` call.

4. **Trace `run_and_generate()` failures to their layer.** This method returns `tuple[str, RagResult]`. If the exception occurs before the LLM call, the `RagResult` will not be available. If it occurs after, inspect the returned result for `elapsed_ms` and `confidence` to understand what the retrieval stage produced.

5. **Check `claim_citations` and `used_native_citations` for generation-side issues.** When `use_native_citations=True` is passed to `run_and_generate()`, the pipeline delegates citation extraction to the LLM. If `used_native_citations` is `False` on the result despite being requested, the provider did not support it — confirm your provider and model support native citations.

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
