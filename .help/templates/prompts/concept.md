---
type: concept
feature: prompts
depth: concept
generated_at: 2026-04-23T03:34:27.946399+00:00
source_hash: 3c35e8e0b30791a15e2c36afa2edf13ea407da219b38803d3325fc50b9980c18
status: generated
---

# Prompts

The prompts module assembles LLM prompts by combining user queries with retrieved documentation passages as grounding context.

## How prompt assembly works

When you ask a question, the system retrieves relevant documentation and packages it with your query into a structured prompt that helps the LLM give accurate, grounded answers.

The process has two stages:

1. **Context preparation** — Retrieved passages are wrapped in `<passage>` tags with injection defense to prevent prompt manipulation
2. **Prompt assembly** — Your query and the prepared context are combined using one of three prompt variants: baseline, citation, or grounded

## Context formatting

The module provides two ways to format retrieved content:

- **`join_context()`** wraps each passage in `<passage>` tags with separator spacing
- **`join_context_numbered()`** adds reference numbers like `[P1]`, `[P2]` for citation-style responses

Both functions respect character limits (configurable via `max_chars`) and include injection defense clauses that instruct the LLM to treat passage content as documentation, not as commands.

## Prompt variants

`build_augmented_prompt()` supports multiple prompt styles through its `variant` parameter:

- **baseline** — Standard question-answering format
- **citation** — Includes instructions for citing sources with reference numbers
- **grounded** — Emphasizes staying within the provided documentation

The function validates that queries are non-empty strings and throws a `ValueError` for unknown variants.

## Security considerations

All context formatting includes an injection defense clause that explicitly tells the LLM to ignore any text within passage tags that looks like instructions or system messages. This prevents retrieved documentation from manipulating the LLM's behavior.
