---
type: warning
name: editor-warning
feature: editor
depth: warning
generated_at: 2026-05-20T03:31:44.071610+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Editor cautions

## What to watch for

The `editor` module provides headless primitives for schema validation, linting, autocomplete, reference lookup, and cross-template rename. These functions operate over a `CorpusProtocol` and are called live by both `attune-gui` and the `attune-author edit` CLI, so unexpected behavior affects interactive users in real time. The risks below are specific to how each subsystem handles edge cases, stale state, and destructive operations.

## Risk areas

### `apply_rename` writes to disk and cannot partially succeed cleanly

`apply_rename` moves files and rewrites template content in place. If the corpus has no resolvable root path, it raises `RenameError` immediately. If a move source is missing at apply time, or if a file has drifted from the planned base since `plan_rename` was called, it also raises `RenameError` — but by that point some edits may already be written. Always call `plan_rename` and `apply_rename` in close succession, and treat any `RenameError` or `RenameCollisionError` as a signal to rebuild the plan from scratch rather than retry the existing one.

### `plan_rename` does not check for name collisions — `apply_rename` does

`RenameCollisionError` is raised during `apply_rename`, not during `plan_rename`. If you display the plan to a user for confirmation before applying it, they may approve a plan that will fail on the collision check. To surface collisions earlier, inspect the `RenamePlan.moves` list and verify that no `FileMove.new_path` already exists in the corpus before presenting the plan.

### `parse_frontmatter` raises `SchemaError` on malformed YAML — it does not return issues

`parse_frontmatter` returns a `(dict, list[FrontmatterIssue])` tuple for structurally valid YAML that fails schema checks. However, if the YAML itself cannot be parsed, it raises `SchemaError` instead of returning an empty dict with issues. If you call `parse_frontmatter` without a `try/except SchemaError` block, a single malformed template will crash the calling code. Wrap the call and present the exception message as a diagnostic rather than letting it propagate.

### `lint_template` line and column positions are 1-indexed

`Diagnostic` fields `line`, `col`, `end_line`, and `end_col` are 1-indexed. Most editor protocol integrations (Language Server Protocol, Monaco) also use 1-based positions, but if you convert diagnostics to a 0-indexed representation — for array slicing or custom UI rendering — subtract 1 from all four fields. Mixing indexing conventions silently highlights the wrong character range.

### `find_references` raises `ValueError` on unsupported `ReferenceKind` values

Passing a `kind` value that `find_references` does not handle raises `ValueError: Unsupported reference kind: {...}` rather than returning an empty list. The same applies to `plan_rename`. If your calling code constructs a `ReferenceKind` dynamically (for example, from a user selection or a serialized string), validate it against the known members of `ReferenceKind` before passing it in.

### Autocomplete results are capped at 50 by default and are prefix-only

`autocomplete_tags` and `autocomplete_aliases` both default to `limit=50` and match only on prefix. A query that does not share a prefix with any tag or alias returns an empty list — it does not fall back to substring or fuzzy matching. If the corpus is large and users expect fuzzy results, you need to implement that layer above these functions.

## How to avoid problems

1. **Separate plan from apply.** Always inspect or log the `RenamePlan` (via `RenamePlan.to_dict()`) before calling `apply_rename`. If the corpus could have changed between the two calls, rebuild the plan immediately before applying.

2. **Guard `parse_frontmatter` with `SchemaError` handling.** Treat a `SchemaError` as a fatal diagnostic for that template and continue processing the rest of the corpus rather than letting the exception terminate a batch operation.

3. **Validate `ReferenceKind` inputs at the boundary.** When accepting a kind from external input, check it against `ReferenceKind` members before passing it to `find_references` or `plan_rename`. This prevents `ValueError` from surfacing as an unhandled exception in the editor UI.

4. **Keep `plan_rename` and `apply_rename` calls close together.** File drift — detected by `apply_rename` as `'File {...} drifted from the planned base; rebuild plan.'` — occurs when templates are edited between planning and application. Minimize that window, especially in contexts where auto-save is active.

5. **Depend only on the public API.** `_PATH_KEYED_SIDECARS` and other underscore-prefixed names can change without notice. Use only the names listed in `__all__` to avoid breakage during refactors.

## Source files

- `src/attune_rag/editor/__init__.py`
- `src/attune_rag/editor/schema.py`
- `src/attune_rag/editor/lint.py`
- `src/attune_rag/editor/autocomplete.py`
- `src/attune_rag/editor/references.py`
- `src/attune_rag/editor/rename.py`

**Tags:** `editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
