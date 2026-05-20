---
type: troubleshooting
name: prompts-troubleshooting
feature: prompts
depth: troubleshooting
generated_at: 2026-05-20T03:24:52.015748+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Troubleshoot prompts

## Before you start

`build_augmented_prompt` assembles a grounded LLM prompt by combining a user query with retrieved passages wrapped in `<passage>...</passage>` sentinels. The `variant` argument selects the prompt style; invalid values raise immediately. If context assembly is the issue, the problem is more likely in `join_context` or `join_context_numbered` than in the prompt renderer itself.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `ValueError: query must be a non-empty string` | The `query` argument passed to `build_augmented_prompt` — it is `None`, `""`, or a non-string type. |
| `ValueError: unknown prompt variant {...}; valid: {...}` | The `variant` argument passed to `build_augmented_prompt` — the error message lists the accepted values. |
| Context block is empty or missing from the rendered prompt | Whether `join_context` or `join_context_numbered` received a non-empty `hits` iterable; the iterable may be exhausted before being passed in. |
| Passages appear without `[P1]`/`[P2]` labels | You are calling `join_context` instead of `join_context_numbered`. The numbered variant wraps each hit in `<passage>[Pn] …</passage>` form. |
| Context is truncated unexpectedly | The `max_chars` argument — it defaults to `DEFAULT_MAX_CONTEXT_CHARS`. Passages are cut off once the concatenated output reaches that limit. |
| Injected text appears to override the system prompt | The `_INJECTION_DEFENSE_CLAUSE` is missing from the rendered prompt, which means the wrong variant or a custom template path is being used. |

## Step-by-step diagnosis

1. **Reproduce the failure with a minimal call.**
   Strip the call to its required arguments and confirm the failure still occurs:

   ```python
   from attune_rag.prompts import build_augmented_prompt
   print(build_augmented_prompt(query="test", context="test context"))
   ```

   If this succeeds, the problem is in how your application builds `query` or `context` before passing them in.

2. **Check the ValueError message.**
   Both `ValueError` cases in `build_augmented_prompt` include the offending value in the message. Read it before going further — it names the bad input directly.

3. **Inspect the context string before it reaches `build_augmented_prompt`.**
   If the rendered prompt is missing passages, print the output of `join_context` or `join_context_numbered` separately:

   ```python
   from attune_rag.prompts import join_context_numbered
   ctx = join_context_numbered(hits=your_hits)
   print(repr(ctx))
   ```

   Confirm the output contains `<passage>` and `</passage>` sentinels and is not an empty string.

4. **Verify the `hits` iterable is not exhausted.**
   Generators in Python can only be consumed once. If `hits` is a generator that was already iterated (for example, for logging), it will be empty when passed to `join_context`. Materialize it first:

   ```python
   hits = list(hits)  # consume once, reuse freely
   ```

5. **Check `max_chars` if context is shorter than expected.**
   The default cap is `DEFAULT_MAX_CONTEXT_CHARS`. If your passages are large, pass a higher limit explicitly:

   ```python
   join_context(hits=hits, max_chars=8000)
   ```

6. **Run the prompts tests.**
   ```bash
   pytest -k "prompts" -v
   ```
   A failing test that exercises your code path will show you the expected inputs and outputs and give you a fixture to work against.

## Common fixes

- **Empty or `None` query:** Validate the query before calling `build_augmented_prompt`:
  ```python
  if not query or not isinstance(query, str):
      raise ValueError(f"Expected a non-empty string, got {query!r}")
  ```

- **Invalid `variant`:** The error message from `build_augmented_prompt` lists valid variant names. Pass one of those exactly — the argument is case-sensitive.

- **Exhausted hits iterable:** Wrap the iterable in `list()` before passing it to `join_context` or `join_context_numbered` (see step 4 above).

- **Context truncated too aggressively:** Pass a larger `max_chars` value to `join_context` or `join_context_numbered`. Note that a very large context may exceed your LLM's token limit — this is a trade-off you control at the call site, not within the prompts module.

- **Prompt injection defense missing:** If you are constructing `context` strings manually and bypassing `join_context_numbered`, the `_INJECTION_DEFENSE_CLAUSE` will not be present. Use `join_context_numbered` so that passages are wrapped correctly and the defense clause can be applied by the prompt template.

## Source files

- `src/attune_rag/prompts.py`

**Tags:** `prompts`, `templates`, `augmentation`, `citation`
