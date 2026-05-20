---
type: tip
name: editor-tip
feature: editor
depth: tip
generated_at: 2026-05-20T03:31:44.081445+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Tip: Separate planning from applying when you rename

## Recommendation

Call `plan_rename()` first, inspect the `RenamePlan` it returns, and only call `apply_rename()` once you are satisfied with the proposed `FileEdit` hunks and `FileMove` entries.

**Why it sticks:** `apply_rename()` writes to disk and refreshes the corpus in one step — there is no dry-run flag, so the plan object is your only preview window.

**Tradeoff:** The plan reflects the corpus state at the moment you called `plan_rename()`. If the corpus changes between planning and applying — for example, another save flushes new templates — `apply_rename()` raises `RenameError: File {...} drifted from the planned base; rebuild plan.` You will need to rebuild the plan. Keep the gap between the two calls short.

## How to use it

1. Compute the plan:

   ```python
   from attune_rag.editor import plan_rename, ReferenceKind

   plan = plan_rename(corpus, old="my-alias", new="better-alias", kind=ReferenceKind.ALIAS)
   ```

2. Review what will change before committing:

   ```python
   import json
   print(json.dumps(plan.to_dict(), indent=2))
   ```

   `plan.edits` lists every `FileEdit` with its diff `hunks`; `plan.moves` lists any `FileMove` path changes.

3. Apply only when the plan looks correct:

   ```python
   from attune_rag.editor import apply_rename

   changed_paths = apply_rename(corpus, plan)
   ```

   `apply_rename()` returns the list of paths it touched. If a `RenameCollisionError` is raised, the proposed `new` name already exists somewhere in the corpus — pick a different name and rebuild the plan.

## Source files

- `src/attune_rag/editor/rename.py`
- `src/attune_rag/editor/__init__.py`

**Tags:** `editor`, `rename`, `refactor`, `template`
