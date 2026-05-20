---
type: warning
name: pipeline-warning
feature: pipeline
depth: warning
generated_at: 2026-05-20T03:20:40.530529+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Pipeline cautions

## What to watch for

`RagPipeline` coordinates corpus retrieval, prompt assembly, and optional LLM generation in a single call. The sections below describe the most consequential pitfalls when integrating or extending the pipeline.

## Risk areas

### Silent fallback when no context is retrieved

If `RagPipeline.run()` finds no matching documents for a query, it switches to `FALLBACK_PROMPT_TEMPLATE` and sets `RagResult.fallback_used = True`. The pipeline does not raise an exception. If you don't check `fallback_used` in your calling code, you may silently serve a fallback response — one that explicitly tells the LLM to admit ignorance — as if it were a grounded answer.

**Mitigation:** Always inspect `result.fallback_used` after `run()` or `run_and_generate()` and handle fallback responses separately in your application logic.

### `confidence` and `elapsed_ms` are informational, not guaranteed

`RagResult.confidence` reflects a retrieval-time signal, not a post-generation quality score. Treating it as a reliability guarantee for the LLM's answer will lead to over-trusting low-quality generations. Similarly, `elapsed_ms` covers pipeline execution time and does not include network latency from the LLM provider.

**Mitigation:** Use `confidence` only for relative ranking or logging, not as a quality gate for user-facing output.

### Native citations change the `RagResult` shape

When you call `run_and_generate()` with `use_native_citations=True`, the pipeline populates `RagResult.claim_citations` and sets `used_native_citations = True`. When `use_native_citations=False` (the default), `claim_citations` is an empty tuple. Code that iterates over `claim_citations` unconditionally will behave differently depending on this flag, with no error to signal the difference.

**Mitigation:** Gate any `claim_citations` processing on `result.used_native_citations` rather than assuming it is always populated.

### The prompt cache splits on a sentinel string

Internally, the pipeline uses `_CACHE_SPLIT = '\n### USER REQUEST\n'` to delimit cached prompt sections. If your query text or corpus content contains this exact string, cache reconstruction will produce a malformed prompt.

**Mitigation:** Sanitize or reject inputs that contain `\n### USER REQUEST\n` before passing them to `run()` or `run_and_generate()`.

### `corpus` is a read-only property after construction

`RagPipeline.corpus` is a property with no setter. Attempting to replace the corpus after instantiation will raise an `AttributeError`. The corpus, retriever, expander, and reranker are all fixed at `__init__` time; build a new `RagPipeline` instance if you need different components.

**Mitigation:** Treat `RagPipeline` instances as immutable after construction and instantiate a new pipeline when component configuration changes.

## How to avoid problems

1. **Check `fallback_used` on every result.** Add an assertion or branch in your integration tests that verifies grounded and fallback code paths are both exercised.

2. **Depend only on the public API.** `_CACHE_SPLIT` and other underscore-prefixed names are implementation details that can change between versions without notice. Depend on the names listed in `__all__` only.

3. **Pin or test against the installed version.** The module exposes `__version__ = '0.1.19'`. If your application has version-specific behavior, assert the version in your test suite so upgrades are caught explicitly.

4. **Isolate environment state between tests.** Module-level configuration and LLM provider settings resolved at import time can make pipeline behavior differ between local and CI environments. Prefer explicit dependency injection over environment variables where possible.

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
