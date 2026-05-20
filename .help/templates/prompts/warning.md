---
type: warning
name: prompts-warning
feature: prompts
depth: warning
generated_at: 2026-05-20T03:24:52.013341+00:00
source_hash: eb6d61656b11230b111f643d8856103251dedb5b5d717c16d6107954b12867f6
status: generated
---

# Prompts cautions

## What to watch for

`build_augmented_prompt` assembles a final LLM prompt by injecting retrieved passages as grounding context. The prompt style is controlled by the `variant` parameter, which must exactly match one of the supported variant names. Passing an unrecognized variant or an empty query raises a `ValueError` at call time — there is no silent fallback.

Retrieved passages are wrapped in `<passage>...</passage>` sentinel tags. The module embeds an injection-defense clause (`_INJECTION_DEFENSE_CLAUSE`) in the prompt to instruct the model to treat passage contents as documentation, never as instructions. If you bypass this wrapping — for example, by concatenating raw hit text yourself — that defense is absent and retrieved content can influence model behavior in unintended ways.

## Risk areas

### Unrecognized prompt variant silently breaks the pipeline

`build_augmented_prompt` raises `ValueError: unknown prompt variant {...}; valid: {...}` when `variant` does not match a supported name. Because `variant` defaults to `'baseline'`, a typo in a caller that overrides the default fails loudly only at runtime, not at import time. Validate the variant string at configuration load time, before any retrieval work has been done.

### Empty query raises immediately

Passing an empty string for `query` raises `ValueError: query must be a non-empty string`. Guard call sites that construct the query dynamically — for example, from user input or a pipeline step that may produce an empty result — with an explicit non-empty check before calling `build_augmented_prompt`.

### Context truncation in `join_context` and `join_context_numbered` silently drops passages

Both `join_context` and `join_context_numbered` enforce a `max_chars` ceiling (defaulting to `DEFAULT_MAX_CONTEXT_CHARS`). Hits that exceed the budget are silently omitted — the functions do not raise an error or return a warning flag. If your retrieval pipeline returns many or large hits, the model may see far fewer passages than you expect. Log the length of the returned context string when debugging retrieval quality issues.

### Passage tags in hit content break the injection defense

`join_context_numbered` wraps each passage in `<passage>...</passage>` tags. If a retrieved document itself contains a literal `</passage>` string, the tag structure is corrupted and the injection-defense clause loses its boundary. The module's `_INJECTION_DEFENSE_CLAUSE` text instructs the model to treat such content as documentation, but that instruction depends on the outer tags remaining intact. Sanitize or escape `</passage>` occurrences in hit content before passing hits to either join function.

### Private constants are not part of the public API

The sentinel strings (`_OPENER`, `_CLOSER`, `_SEPARATOR`) and the injection-defense clause (`_INJECTION_DEFENSE_CLAUSE`) are underscore-prefixed and may change without notice. If your code references these directly — for example, to parse or post-process a prompt string — it may break on any update to the module. Use `join_context` or `join_context_numbered` to produce sentinel-wrapped context rather than constructing it manually.

## How to avoid problems

1. **Validate the variant at startup.** Pass `variant` through `build_augmented_prompt` with a known-good test query during application initialization so that misconfiguration fails fast rather than mid-request.

2. **Guard against empty queries.** Check `query.strip()` before calling `build_augmented_prompt`. An empty string will always raise; a whitespace-only string may produce a degenerate prompt depending on the variant.

3. **Log context length after joining.** After calling `join_context` or `join_context_numbered`, log `len(context)` alongside the number of input hits. A large gap between hits provided and characters returned indicates truncation.

4. **Use the provided join functions instead of manual concatenation.** Building context strings by hand omits the `<passage>` sentinel wrapping and the injection-defense clause, removing the prompt-injection mitigation the module provides.

5. **Run targeted tests before changing variants or context logic.** `pytest -k "prompts"` catches regressions in prompt structure before they affect the rest of the pipeline.

## Source files

- `src/attune_rag/prompts.py`

**Tags:** `prompts`, `templates`, `augmentation`, `citation`
