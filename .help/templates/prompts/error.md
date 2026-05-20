---
type: error
name: prompts-error
feature: prompts
depth: error
generated_at: 2026-05-20T03:24:52.007891+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Prompts errors

## Common error signatures

Errors in this module come from invalid inputs to `build_augmented_prompt`. Both are `ValueError`:

- **`query must be a non-empty string`** — raised when `query` is an empty string or a non-string type. Check that the value passed to `build_augmented_prompt(query=...)` is a non-empty `str` before calling the function.
- **`unknown prompt variant {...}; valid: {...}`** — raised when the `variant` argument does not match any registered prompt variant. The error message includes both the unrecognised value you passed and the set of accepted values, so you can correct the call or register a new variant.

## Where errors originate

- `build_augmented_prompt(query, context, variant)` — validates `query` and `variant` before rendering; raises `ValueError` for either input constraint listed above.
- `join_context(hits, corpus, max_chars)` — concatenates `RetrievalHit` contents into `<passage>...</passage>`-wrapped context strings; errors here typically indicate problems with the `hits` iterable or a `corpus` that does not satisfy `CorpusProtocol`.
- `join_context_numbered(hits, corpus, max_chars)` — same wrapping behaviour as `join_context`, but produces `[P1]`/`[P2]`-labelled passage bodies; the same iterable and corpus constraints apply.

## How to diagnose

1. **Read the `ValueError` message directly.** Both validation errors in `build_augmented_prompt` include the offending value and, for variant mismatches, the full list of valid options. You usually do not need a debugger — fix the argument shown in the message.

2. **Confirm `query` is a non-empty string before the call.** If `query` is assembled dynamically (e.g. from user input or a retrieval pipeline), add an assertion or guard upstream:
   ```python
   if not isinstance(query, str) or not query:
       raise ValueError("query must be provided before calling build_augmented_prompt")
   ```

3. **Check the `variant` value against the valid set.** When the unknown-variant error fires, the message prints both the bad value and the accepted variants. Verify that any string passed as `variant` exactly matches one of those values (the comparison is case-sensitive).

4. **Inspect the `hits` iterable when context is empty or malformed.** If the rendered prompt contains no passage content, confirm that `join_context` or `join_context_numbered` is receiving a non-empty iterable of `RetrievalHit` objects. An empty iterable produces an empty context string, which `build_augmented_prompt` will silently embed — no exception is raised, but the LLM receives no grounding passages.

5. **Verify prompt injection defence is not corrupting passage content.** The `<passage>` sentinel wrapper includes an injection-defence clause. If downstream parsing breaks on the rendered prompt, check whether your hit contents themselves contain literal `</passage>` strings, which the module deliberately treats as documentation rather than closing tags.

## Source files

- `src/attune_rag/prompts.py`

**Tags:** `prompts`, `templates`, `augmentation`, `citation`
