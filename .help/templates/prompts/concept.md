---
type: concept
name: prompts-concept
feature: prompts
depth: concept
generated_at: 2026-05-15T20:02:27.251670+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Prompts

The `prompts` module assembles the final text that attune-rag sends to an LLM — combining a user query, retrieved documentation passages, and a prompt variant into a single, grounded request.

## How prompt assembly works

When you call `build_augmented_prompt(query, context, variant)`, the module takes three inputs and renders them into one prompt string:

1. **The query** — a non-empty string representing what the user asked. Passing an empty string raises a `ValueError`.
2. **The context** — a block of retrieved passages produced by `join_context()` or `join_context_numbered()`, wrapped in sentinel tags so the LLM can distinguish documentation from instructions.
3. **The variant** — a named prompt style (such as `baseline`, `citation`, or `grounded`). Passing an unrecognised variant raises a `ValueError` listing the valid options.

The context block itself is built by one of two helpers:

- `join_context()` wraps each retrieved hit in `<passage>...</passage>` tags and concatenates them up to a character limit (`DEFAULT_MAX_CONTEXT_CHARS`). Use this when you need a clean, unmarked block.
- `join_context_numbered()` does the same but labels each passage `[P1]`, `[P2]`, and so on. Use this when the prompt variant needs the LLM to cite specific passages by number.

Both helpers accept an optional `CorpusProtocol` to resolve hit contents, and both silently stop adding passages once the character budget is reached.

## Injection defense

Every assembled context block carries `_INJECTION_DEFENSE_CLAUSE` — a standing instruction that tells the LLM to treat anything inside `<passage>...</passage>` tags as retrieved documentation, never as a command. This means even if a retrieved document contains text that looks like a system message or an attempt to break out of the wrapper, the LLM is instructed to read it as content about that technique, not as a directive to follow.

## How the pieces fit together

```
user query
    │
    ▼
build_augmented_prompt(query, context, variant)
    │                        │
    │              join_context()          ← plain sentinel-wrapped passages
    │              join_context_numbered() ← [P1]/[P2]-labelled passages
    │                        │
    │              <passage>...</passage> blocks
    │              + _INJECTION_DEFENSE_CLAUSE
    │
    ▼
rendered prompt string → LLM
```

The variant controls which prompt template wraps the query and context. Choosing `citation` or `grounded` (versus `baseline`) changes the instructions the LLM receives, but the context-assembly step is the same regardless of variant.
