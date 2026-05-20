---
type: note
name: prompts-note
feature: prompts
depth: note
generated_at: 2026-05-20T03:24:52.025125+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Note: prompts

## Overview

The `prompts` module (`src/attune_rag/prompts.py`) assembles augmented prompts for LLM calls. It wraps retrieved passages in `<passage>…</passage>` sentinels, injects them as grounding context, and renders the final prompt string through `build_augmented_prompt()`.

## How the module works

The module exposes three top-level functions — no class instantiation required:

- **`build_augmented_prompt(query, context, variant)`** — Renders the final prompt. `variant` selects the prompt style; passing an unrecognised value raises `ValueError` listing the valid options. An empty `query` also raises `ValueError`.
- **`join_context(hits, corpus, max_chars)`** — Concatenates `RetrievalHit` contents into a single string, wrapping each passage in `<passage>…</passage>` tags and separating entries with `\n\n`. Truncates at `max_chars`.
- **`join_context_numbered(hits, corpus, max_chars)`** — Same structure as `join_context()`, but prefixes each passage body with a `[P1]`, `[P2]`, … label to support citation-style prompts.

## Prompt injection defence

Every passage block is governed by `_INJECTION_DEFENSE_CLAUSE`, a hardcoded instruction that tells the model to treat content inside `<passage>…</passage>` tags as retrieved documentation only — never as directives or system messages. This means the sentinel tags serve a dual purpose: they delimit context for the prompt renderer *and* carry the injection-defence instruction to the model.

**Tags:** `prompts`, `templates`, `augmentation`, `citation`
