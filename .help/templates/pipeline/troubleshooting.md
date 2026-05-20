---
type: troubleshooting
name: pipeline-troubleshooting
feature: pipeline
depth: troubleshooting
generated_at: 2026-05-20T03:20:40.533028+00:00
source_hash: f5cc845ee3957a76674338c9a162ce4a86e404c42291f721ed77a3b4c3b27569
status: generated
---

# Troubleshoot pipeline

## Overview

`RagPipeline` orchestrates corpus retrieval, prompt assembly, and optional LLM generation in a single call. `RagPipeline.run()` returns a `RagResult` dataclass containing the augmented prompt, `CitationRecord` provenance, a confidence score, and a `fallback_used` flag that indicates whether the pipeline found any grounding context for the query.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `RagResult.fallback_used` is `True` | The corpus returned no hits — verify `RagPipeline.corpus` is populated and that your query terms match indexed content. |
| `RagResult.confidence` is `0.0` or unexpectedly low | Inspect `RagResult.context` (the raw retrieved text). An empty string means retrieval returned nothing; a non-empty string means reranking or scoring is the issue. |
| `augmented_prompt` contains the fallback template | `fallback_used` is `True`. The pipeline substituted `FALLBACK_PROMPT_TEMPLATE`, which instructs the LLM to answer without grounding. Check corpus ingestion. |
| `run_and_generate()` raises on `provider` | The `provider` argument accepts an `LLMProvider` instance or a string name. Confirm the value matches a registered provider. |
| `claim_citations` is an empty tuple | `used_native_citations` is `False` and no claim-level attribution was attempted, or the `prompt_variant` used does not produce claim citations. |
| Intermittent retrieval differences | `QueryExpander` or `LLMReranker` are non-deterministic — check whether the same `k` and `prompt_variant` values were used across runs. |
| `elapsed_ms` is unexpectedly high | The reranker (`LLMReranker`) makes an LLM call during retrieval. Remove it from the pipeline constructor to isolate retrieval latency from generation latency. |

## Step-by-step diagnosis

1. **Reproduce the failure with a minimal call.**
   Strip `RagPipeline` down to its required arguments and call `run()` directly:

   ```python
   from attune_rag import RagPipeline, DirectoryCorpus

   pipeline = RagPipeline(corpus=DirectoryCorpus("path/to/docs"))
   result = pipeline.run("your failing query", k=3)
   print(result)
   ```

   Confirm the failure occurs here before involving `run_and_generate()`, an expander, or a reranker.

2. **Inspect the `RagResult` fields.**
   After calling `run()`, print every field to locate where the pipeline diverges from expectations:

   ```python
   print("fallback_used:", result.fallback_used)
   print("confidence:", result.confidence)
   print("context:", result.context[:500])
   print("augmented_prompt:", result.augmented_prompt[:500])
   print("citation:", result.citation)
   print("claim_citations:", result.claim_citations)
   print("elapsed_ms:", result.elapsed_ms)
   ```

   - `context` empty → the problem is in retrieval, not prompt assembly.
   - `context` populated but `augmented_prompt` wrong → the problem is in prompt assembly (`build_augmented_prompt`, `prompt_variant`).
   - `fallback_used: True` → no corpus hits; see corpus and retriever fixes below.

3. **Enable DEBUG logging.**
   Add the following before your pipeline call to surface internal retrieval and reranking decisions:

   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

   Re-run and look for log lines from `attune_rag.pipeline`.

4. **Isolate optional components.**
   If you passed `expander` or `reranker` to the constructor, remove them and re-run `run()`. Both components transform the query or reorder hits before prompt assembly — removing them confirms whether the root cause is in the core pipeline or in the optional components.

5. **Run the pipeline test suite.**
   ```bash
   pytest -k "pipeline" -v
   ```
   If a test covers the failing path, check whether it passes. A passing test with a failing integration call usually means environment or input data differs from the fixture.

## Common fixes

- **Corpus is empty or not attached.**
  If `RagPipeline.corpus` was not passed to the constructor, `RagResult.fallback_used` will be `True` for every query. Pass a populated corpus at construction time:

  ```python
  pipeline = RagPipeline(corpus=DirectoryCorpus("path/to/docs"))
  ```

- **Wrong `prompt_variant`.**
  `run()` and `run_and_generate()` accept a `prompt_variant` argument (default `'citation'`). Passing an unrecognized variant can produce unexpected prompts. Check the allowed values in `PROMPT_VARIANTS`:

  ```python
  from attune_rag import PROMPT_VARIANTS
  print(PROMPT_VARIANTS)
  ```

- **Retriever returns too few hits.**
  The default `k=3` may be too low if your corpus is large and the relevant document ranks outside the top 3. Increase `k`:

  ```python
  result = pipeline.run("your query", k=10)
  ```

- **Stale corpus cache.**
  If documents were added to the corpus but retrieval still misses them, the corpus index may be stale. Re-instantiate `DirectoryCorpus` (or your custom `CorpusProtocol` implementation) to force re-indexing.

- **Dependency version mismatch.**
  This module is version `0.1.19`. If behavior changed after an update, confirm the installed version:

  ```bash
  pip show attune-rag
  ```

  Pin to `0.1.19` if a newer version introduced a regression:

  ```bash
  pip install "attune-rag==0.1.19"
  ```

- **`use_native_citations` not set.**
  If you expect `used_native_citations: True` in `RagResult` but see `False`, you must pass `use_native_citations=True` explicitly to `run_and_generate()` — it defaults to `False`.

## Source files

- `src/attune_rag/pipeline.py`
- `src/attune_rag/__init__.py`

**Tags:** `pipeline`, `orchestration`, `rag`, `result`
