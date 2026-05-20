---
type: comparison
name: prompts-comparison
feature: prompts
depth: comparison
generated_at: 2026-05-20T03:24:52.027587+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Comparison: `join_context` vs `join_context_numbered`

## Overview

Both functions assemble retrieved passages into a context string that `build_augmented_prompt` can inject into an LLM prompt. They differ in how they label passages and whether they defend against prompt injection.

| Capability | `join_context` | `join_context_numbered` |
|---|---|---|
| Passage wrapping | `<passage>…</passage>` | `<passage>…</passage>` |
| Passage labels | None | `[P1]`, `[P2]`, … |
| Injection-defense clause | Yes (`_INJECTION_DEFENSE_CLAUSE` prepended) | Yes (`_INJECTION_DEFENSE_CLAUSE` prepended) |
| Suitable for citation-style prompts | No — no stable label for the model to cite | Yes — labels give the model a citation target |
| Output when model must explain its sources | Harder; model cannot reference a specific passage | Straightforward; model can write "see [P2]" |
| Character budget | Configurable via `max_chars` | Configurable via `max_chars` |
| Accepts optional `CorpusProtocol` | Yes | Yes |

Both functions wrap every passage in `<passage>…</passage>` tags and prepend `_INJECTION_DEFENSE_CLAUSE`, which instructs the model to treat content inside those tags as documentation rather than executable instructions — even if that content contains directives or attempts to close the wrapping tag prematurely.

## When to use `join_context`

Use `join_context` when:

- The downstream prompt does not need to attribute answers to individual passages — for example, a "grounded" or "baseline" variant built with `build_augmented_prompt`.
- Your retrieved hits are short and numerous, and adding numeric labels would clutter the context without benefit.
- You want the simplest possible context assembly with no labeling overhead.

## When to use `join_context_numbered`

Use `join_context_numbered` when:

- The model's response should cite specific passages (for example, a "citation" variant passed to `build_augmented_prompt`).
- You want the model to reference sources by label (`[P1]`, `[P2]`) so downstream parsing or UI rendering can map citations back to the original hits.
- Transparency and auditability of retrieved evidence matter to your use case.

## When to use neither directly

- If you only need a finished prompt string and do not need to inspect the intermediate context, call `build_augmented_prompt(query, context, variant=...)` directly and let it handle the rendering. Compose the context string first with one of the two functions above, then pass it in.
- If you need a prompt shape that none of the supported variants cover, do not patch the `_INJECTION_DEFENSE_CLAUSE` or sentinel constants — file an issue or propose a new variant rather than bypassing the injection-defense layer.
- If you are writing a throwaway script that will never enter production, assembling a plain f-string is faster than wiring up the full context pipeline.

## Recommendation

Prefer `join_context_numbered` when in doubt. The `[P1]`/`[P2]` labels cost nothing at assembly time and give the model — and any downstream parser — a stable handle on each passage. Switch to `join_context` only when you have a specific reason to omit labels, such as a prompt variant that treats context as a single undifferentiated block.

## Source files

- `src/attune_rag/prompts.py`

**Tags:** `prompts`, `templates`, `augmentation`, `citation`
