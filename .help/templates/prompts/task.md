---
type: task
name: prompts-task
feature: prompts
depth: task
generated_at: 2026-05-15T20:02:27.255457+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Work with prompts

Use the prompts module when you need to assemble a grounded LLM prompt — it combines a user query, retrieved passages, and a prompt variant into a single augmented prompt ready to send to a model.

## Prerequisites

- Access to `src/attune_rag/prompts.py`
- Retrieved passages available as `RetrievalHit` objects (or a plain iterable)

## Steps

1. **Format your retrieved passages into a context string.**
   Call `join_context()` or `join_context_numbered()` to concatenate your `RetrievalHit` objects into a sentinel-wrapped string:

   - Use `join_context()` when your prompt template does not require numbered citation markers.
   - Use `join_context_numbered()` when you need `[P1]`/`[P2]`-style passage labels for citation.

   Both functions accept an optional `corpus` and a `max_chars` cap (default: `DEFAULT_MAX_CONTEXT_CHARS`).

2. **Choose a prompt variant.**
   `build_augmented_prompt()` accepts a `variant` parameter. Valid values are `'baseline'` (the default) and any other variant names defined in the module. Passing an unknown variant raises `ValueError: 'unknown prompt variant {...}; valid: {...}'`.

3. **Build the augmented prompt.**
   Call `build_augmented_prompt()` with your query, the context string from step 1, and your chosen variant:

   ```python
   from attune_rag.prompts import build_augmented_prompt, join_context_numbered

   context = join_context_numbered(hits)
   prompt = build_augmented_prompt(query="What is template migration?", context=context, variant="baseline")
   ```

   Pass a non-empty string as `query`; an empty string raises `ValueError: 'query must be a non-empty string'`.

4. **Run the related tests.**
   Verify that your usage does not break existing behaviour:

   ```
   pytest -k "prompts"
   ```

## Key files

- `src/attune_rag/prompts.py`

## Verify success

`build_augmented_prompt()` returns a non-empty string containing your query, the passage content wrapped in `<passage>...</passage>` sentinels, and the injection-defense clause. Confirm that the returned prompt includes all three before sending it to the model.
