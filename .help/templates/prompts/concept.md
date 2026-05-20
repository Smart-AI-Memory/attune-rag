---
type: concept
name: prompts-concept
feature: prompts
depth: concept
generated_at: 2026-05-20T03:24:51.993011+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Prompts

The `prompts` module assembles augmented prompt templates that combine a user query with retrieved documentation passages before sending them to an LLM.

## Mental model

Retrieval-augmented generation works by inserting retrieved text into the prompt so the LLM can ground its answer in your documentation. The `prompts` module sits between your retrieval results and the LLM call: it takes a list of `RetrievalHit` objects, formats them into a context block, and then renders a complete prompt string that the LLM receives.

The flow looks like this:

```
RetrievalHit list
      │
      ▼
join_context() or join_context_numbered()
      │  wraps each passage in <passage>…</passage>
      ▼
build_augmented_prompt(query, context, variant)
      │  selects a prompt style and injects the context
      ▼
Rendered prompt string → LLM
```

## Passage wrapping and prompt injection defense

Both context-building functions wrap passage content in `<passage>` … `</passage>` sentinel tags. This structure does two things:

1. **Signals source boundaries** — the LLM can clearly distinguish retrieved documentation from the instruction text.
2. **Defends against prompt injection** — every rendered prompt includes a built-in defense clause that instructs the LLM to treat anything inside `<passage>` tags as documentation content, never as instructions, even if that content contains text that looks like directives or system messages.

## Core functions

### `join_context(hits, corpus, max_chars)`

Concatenates the contents of each `RetrievalHit` into a single string, separating passages with `\n\n` and wrapping each one in `<passage>…</passage>` tags. The `max_chars` limit (defaulting to `DEFAULT_MAX_CONTEXT_CHARS`) prevents the context block from exceeding a safe token budget.

### `join_context_numbered(hits, corpus, max_chars)`

Works the same way as `join_context()`, but prefixes each passage with a citation label — `[P1]`, `[P2]`, and so on — inside the `<passage>` wrapper. Use this variant when your prompt style needs the LLM to cite specific passages by number in its response.

### `build_augmented_prompt(query, context, variant)`

Renders the final prompt string from a query, a pre-built context block, and a named variant. The `variant` parameter selects among the supported prompt styles (for example `'baseline'`). The function raises `ValueError` if `query` is empty or if you pass an unrecognized variant name, so callers can catch configuration errors before making an LLM call.

## Prompt variants

The `variant` argument to `build_augmented_prompt()` controls which template structure the LLM receives. Valid variant names are enforced at runtime — passing an unknown name raises:

```
ValueError: unknown prompt variant {…}; valid: {…}
```

This makes it straightforward to add or restrict styles without silent fallbacks.
