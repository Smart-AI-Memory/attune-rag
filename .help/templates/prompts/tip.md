---
type: tip
name: prompts-tip
feature: prompts
depth: tip
generated_at: 2026-05-20T03:24:52.022891+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Tip: Use `join_context_numbered` when your prompt needs citation support

## Recommendation

When you need users or downstream tooling to trace an answer back to a specific
source passage, use `join_context_numbered` instead of `join_context`. It wraps
each passage in `<passage>` tags with a `[P1]`/`[P2]` label that your prompt
can reference directly.

**Why it sticks:** `join_context` produces sentinel-wrapped context, but the
numbered variant gives each passage an address — without that address, citations
in the generated answer have nothing concrete to point at.

## Tradeoff

Numbered passages add a small amount of token overhead (`[P1]`, `[P2]`, etc.)
for every hit. For very large context windows with many hits, that overhead is
negligible. For tight token budgets, use `join_context` instead and accept that
you lose per-passage traceability.

## Details

Both helpers accept the same signature:

```python
join_context(hits, corpus=None, max_chars=DEFAULT_MAX_CONTEXT_CHARS)
join_context_numbered(hits, corpus=None, max_chars=DEFAULT_MAX_CONTEXT_CHARS)
```

Pass the result directly to `build_augmented_prompt` as the `context` argument:

```python
context = join_context_numbered(hits)
prompt = build_augmented_prompt(query=query, context=context, variant="baseline")
```

Note that `build_augmented_prompt` raises `ValueError` for an empty `query` and
for any unrecognized `variant` value, so validate those inputs before calling it.

## Source files

- `src/attune_rag/prompts.py`

**Tags:** `prompts`, `templates`, `augmentation`, `citation`
