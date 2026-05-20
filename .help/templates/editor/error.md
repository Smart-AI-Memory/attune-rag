---
type: error
name: editor-error
feature: editor
depth: error
generated_at: 2026-05-20T03:31:44.066050+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Editor errors

## Common error signatures

The editor feature raises errors across four areas: frontmatter parsing, lint diagnostics, reference lookup, and rename execution. The following exceptions are the most common failure modes.

**`SchemaError`** — raised by `parse_frontmatter()` when the frontmatter block cannot be parsed as YAML at all:

```
SchemaError: Malformed YAML in frontmatter: <detail>
```

**`ValueError`** — raised by `find_references()` or `plan_rename()` when the caller passes an unsupported `ReferenceKind`:

```
ValueError: Unsupported reference kind: <kind>
ValueError: Unsupported rename kind: <kind>
```

**`RenameCollisionError`** — a subclass of `RenameError`, raised by `apply_rename()` when the proposed new name already exists in the corpus. Carries the conflicting `name` and the `owning_path` of the file that owns it.

**`RenameError`** — the base class for all rename failures, raised by `apply_rename()` in several distinct states:

```
RenameError: Corpus has no resolvable root path; apply is not supported.
RenameError: Move source missing at apply time: <path>
RenameError: Failed to move <old> -> <new>: <detail>
RenameError: Planned target does not exist: <path>
RenameError: File <path> drifted from the planned base; rebuild plan.
```

## How to diagnose

### Frontmatter and schema errors

- If you see `SchemaError`, the YAML in the frontmatter block is structurally invalid — not just semantically wrong. Check for unclosed quotes, invalid indentation, or duplicate keys.
- If `parse_frontmatter()` returns without raising but reports `FrontmatterIssue` entries, the YAML parsed successfully but failed schema validation. Each `FrontmatterIssue` includes a `code`, a `message`, and a `path` tuple pointing to the offending key. Inspect those fields directly.
- You can validate already-parsed frontmatter independently with `validate_frontmatter(data)` to isolate whether the problem is in parsing or in the schema contract.

### Reference and rename kind errors

- `ValueError: Unsupported reference kind` means the `kind` argument passed to `find_references()` or `plan_rename()` is not a valid `ReferenceKind` enum member. Verify that the caller is using a value from the `ReferenceKind` enum, not a raw string.

### Rename application errors

`apply_rename()` applies a `RenamePlan` to disk. It can fail at several points:

- **`Corpus has no resolvable root path`** — the corpus does not expose a filesystem root, so `apply_rename()` cannot write files. This is a configuration issue, not a bug in the plan itself. Use a corpus backed by a real directory.
- **`Move source missing at apply time`** — the file that the plan expects to move was not found on disk when `apply_rename()` ran. The file may have been deleted or renamed between `plan_rename()` and `apply_rename()`. Rebuild the plan.
- **`RenameCollisionError`** — a file already exists at the new path. Check `error.name` for the conflicting name and `error.owning_path` for its location. Either choose a different name or remove the conflict first.
- **`Failed to move <old> -> <new>`** — the filesystem move itself failed. The detail string contains the underlying OS error. Check permissions and whether the destination directory exists.
- **`Planned target does not exist`** — after the move, the expected destination file was not present. This can indicate a filesystem issue or a mismatch between the plan and the actual directory layout.
- **`File drifted from the planned base; rebuild plan`** — the file's content changed between when `plan_rename()` computed the plan and when `apply_rename()` attempted to apply it. Call `plan_rename()` again on the current corpus state and retry.

### Lint diagnostics

`lint_template()` does not raise on lint failures — it returns a list of `Diagnostic` objects. If you expect diagnostics but receive an empty list, confirm that `text` is non-empty and that `rel_path` resolves correctly within the corpus. Each `Diagnostic` carries a `severity`, `code`, `message`, and 1-indexed `line`/`col`/`end_line`/`end_col` range.

## Source files

- `src/attune_rag/editor/__init__.py`
- `src/attune_rag/editor/schema.py`
- `src/attune_rag/editor/lint.py`
- `src/attune_rag/editor/autocomplete.py`
- `src/attune_rag/editor/references.py`
- `src/attune_rag/editor/rename.py`

**Tags:** `editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
