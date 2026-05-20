---
type: faq
name: prompts-faq
feature: prompts
depth: faq
generated_at: 2026-05-20T03:24:52.018126+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Prompts FAQ

## What does the prompts module do?

It provides helpers for assembling augmented prompts: `build_augmented_prompt` combines your query with retrieved passages to produce a grounded LLM prompt, and `join_context` / `join_context_numbered` format those passages into sentinel-wrapped context strings.

## When should I use it?

Use this module when you need to construct a retrieval-augmented prompt — that is, when you have a user query and a set of retrieved passages that you want to inject as grounding context before sending to an LLM. If you're not doing retrieval-augmented generation, this module probably isn't what you need.

## What's the main entry point?

Start with `build_augmented_prompt(query, context, variant)`. It takes your query string and a pre-built context string, and returns a fully rendered prompt.

To build the context string from retrieval hits, use one of:

- `join_context(hits, corpus, max_chars)` — wraps each passage in `<passage>...</passage>` tags.
- `join_context_numbered(hits, corpus, max_chars)` — same wrapping, but labels passages `[P1]`, `[P2]`, etc., which is useful when your prompt needs to cite specific passages by number.

## What prompt variants are available?

`build_augmented_prompt` accepts a `variant` parameter. Pass an unrecognized value and it raises a `ValueError` that lists the valid variant names — for example:

```
ValueError: unknown prompt variant 'foo'; valid: {'baseline', ...}
```

The default is `'baseline'`.

## What happens if I pass an empty query?

`build_augmented_prompt` raises a `ValueError` with the message `'query must be a non-empty string'`. Validate your query before calling the function if your input source can produce empty strings.

## How does the module guard against prompt injection?

Retrieved passages are wrapped in `<passage>...</passage>` tags, and an injection-defense clause is injected into the prompt. That clause instructs the LLM to treat anything inside the tags as documentation content, never as instructions — even if the passage text contains directives or attempts to close the `</passage>` tag early.

## How do I debug problems?

Run the module's tests first:

```
pytest -k "prompts" -v
```

If the tests pass but your code still fails, the most common causes are an empty query string (raises `ValueError`) or an unrecognized `variant` value (also raises `ValueError` with a list of valid options). Check the exception message — it tells you exactly what went wrong.

## Where is the source code?

`src/attune_rag/prompts.py`

**Tags:** `prompts`, `templates`, `augmentation`, `citation`
