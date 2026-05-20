---
type: task
name: prompts-task
feature: prompts
depth: task
generated_at: 2026-05-20T03:24:51.999072+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Build augmented prompts for an LLM

Use the `prompts` module when you need to assemble a grounded prompt from retrieved passages before sending a query to an LLM.

## Prerequisites

- Python access to `src/attune_rag/prompts.py`
- A list of `RetrievalHit` objects to use as grounding context

## Build and send an augmented prompt

1. **Format your retrieved hits into a context string.**
   Call `join_context()` to concatenate hit contents into a single sentinel-wrapped string, or call `join_context_numbered()` if you want each passage labelled `[P1]`, `[P2]`, and so on inside `<passage>` tags:

   ```python
   from attune_rag.prompts import join_context, join_context_numbered

   # Plain sentinel-wrapped context
   context = join_context(hits)

   # Numbered passage context
   context = join_context_numbered(hits)
   ```

   Both functions accept an optional `corpus` and a `max_chars` limit to cap total context size.

2. **Build the augmented prompt.**
   Pass your query string and the context string to `build_augmented_prompt()`. Supply a `variant` argument to select a prompt style; the default is `'baseline'`:

   ```python
   from attune_rag.prompts import build_augmented_prompt

   prompt = build_augmented_prompt(query=query, context=context, variant="baseline")
   ```

   - `query` must be a non-empty string; otherwise a `ValueError` is raised.
   - If you pass an unrecognised `variant`, a `ValueError` lists the valid options.

3. **Send the rendered prompt to your LLM.**
   Pass the string returned by `build_augmented_prompt()` directly to your model's API. The prompt already includes the injection-defence clause that instructs the model to treat content inside `<passage>…</passage>` tags as documentation, not as instructions.

4. **Run the prompt-related tests.**
   Verify your integration with:

   ```bash
   pytest -k "prompts"
   ```

## Verify success

The task succeeded when:

- `build_augmented_prompt()` returns a non-empty string containing your query and the sentinel-wrapped passages.
- All `pytest -k "prompts"` tests pass with no errors or failures.

## Key files

- `src/attune_rag/prompts.py`
