---
type: reference
name: editor-reference
feature: editor
depth: reference
generated_at: 2026-05-20T02:44:35.490536+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Editor reference

Headless primitives for schema validation, linting, autocomplete, reference lookup, and cross-template rename refactoring. Use these functions and dataclasses directly from `attune-rag` to power editor tooling or the `attune-author edit` CLI without loading a GUI.

## Classes

| Class | Description |
|-------|-------------|
| `Diagnostic` | A single lint diagnostic. |
| `Reference` | A single reference to a name in a corpus. |
| `RenameError` | Base class for rename refactor failures. |
| `RenameCollisionError` | Raised when the proposed new name already exists. |
| `Hunk` | A single unified-diff hunk. |
| `FileEdit` | A planned edit to a single template file. |
| `FileMove` | A planned file move (rename of a template's rel-path). |
| `RenamePlan` | A complete rename plan, containing per-file edits and file moves. |
| `SchemaError` | Raised when the frontmatter cannot be parsed at all (malformed YAML). |
| `FrontmatterIssue` | A single schema violation. Line and column are 1-indexed within the frontmatter block. |

### `Diagnostic` fields

| Field | Type | Default |
|-------|------|---------|
| `severity` | `Severity` | — |
| `code` | `str` | — |
| `message` | `str` | — |
| `line` | `int` | — |
| `col` | `int` | — |
| `end_line` | `int` | — |
| `end_col` | `int` | — |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict(self)` | `dict[str, Any]` | Serialize this diagnostic to a plain dictionary. |

### `Reference` fields

| Field | Type | Default |
|-------|------|---------|
| `template_path` | `str` | — |
| `line` | `int` | — |
| `col` | `int` | — |
| `context` | `ReferenceContext` | — |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict(self)` | `dict[str, Any]` | Serialize this reference to a plain dictionary. |

### `Hunk` fields

| Field | Type | Default |
|-------|------|---------|
| `hunk_id` | `str` | — |
| `header` | `str` | — |
| `lines` | `list[str]` | `field(default_factory=list)` |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict(self)` | `dict[str, Any]` | Serialize this hunk to a plain dictionary. |

### `FileEdit` fields

| Field | Type | Default |
|-------|------|---------|
| `path` | `str` | — |
| `old_text` | `str` | — |
| `new_text` | `str` | — |
| `hunks` | `list[Hunk]` | `field(default_factory=list)` |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict(self)` | `dict[str, Any]` | Serialize this file edit to a plain dictionary. |

### `FileMove` fields

| Field | Type | Default |
|-------|------|---------|
| `old_path` | `str` | — |
| `new_path` | `str` | — |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict(self)` | `dict[str, Any]` | Serialize this file move to a plain dictionary. |

### `RenamePlan` fields

| Field | Type | Default |
|-------|------|---------|
| `old` | `str` | — |
| `new` | `str` | — |
| `kind` | `ReferenceKind` | — |
| `edits` | `list[FileEdit]` | `field(default_factory=list)` |
| `moves` | `list[FileMove]` | `field(default_factory=list)` |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `to_dict(self)` | `dict[str, Any]` | Serialize this rename plan to a plain dictionary. |

### `FrontmatterIssue` fields

| Field | Type | Default |
|-------|------|---------|
| `code` | `str` | — |
| `message` | `str` | — |
| `path` | `tuple[str \| int, ...]` | `()` |

### `RenameCollisionError`

```
RenameCollisionError(name: str, owning_path: str) -> None
```

Raised when the proposed new name already exists. Subclass of `RenameError`.

## Functions

| Function | Parameters | Returns | Description |
|----------|------------|---------|-------------|
| `autocomplete_tags` | `corpus: Any, prefix: str, limit: int = 50` | `list[str]` | Return up to `limit` tag suggestions starting with `prefix`. |
| `autocomplete_aliases` | `corpus: Any, prefix: str, limit: int = 50` | `list[AliasInfo]` | Return up to `limit` alias suggestions starting with `prefix`. |
| `lint_template` | `text: str, rel_path: str, corpus: Any` | `list[Diagnostic]` | Run all lint checks against `text`. |
| `find_references` | `corpus: Any, name: str, kind: ReferenceKind` | `list[Reference]` | Return every reference to `name` across the corpus. |
| `plan_rename` | `corpus: Any, old: str, new: str, kind: ReferenceKind` | `RenamePlan` | Compute a `RenamePlan` for renaming `old` to `new`. |
| `apply_rename` | `corpus: Any, plan: RenamePlan` | `list[str]` | Apply `plan` to disk and refresh the corpus. |
| `load_schema` | — | `dict[str, Any]` | Return the parsed JSON Schema for template frontmatter. |
| `validate_frontmatter` | `data: Any` | `list[FrontmatterIssue]` | Validate already-parsed frontmatter against the schema. |
| `parse_frontmatter` | `yaml_text: str` | `tuple[dict[str, Any], list[FrontmatterIssue]]` | Parse a YAML frontmatter block and validate it. |

### Raises

#### `find_references`

| Raises | Message |
|--------|---------|
| `ValueError` | `'Unsupported reference kind: {...}'` |

#### `plan_rename`

| Raises | Message |
|--------|---------|
| `ValueError` | `'Unsupported rename kind: {...}'` |

#### `apply_rename`

| Raises | Message |
|--------|---------|
| `RenameError` | `'Corpus has no resolvable root path; apply is not supported.'` |
| `RenameError` | `'Move source missing at apply time: {...}'` |
| `RenameCollisionError` | — |
| `RenameError` | `'Failed to move {...} -> {...}: {...}'` |
| `RenameError` | `'Planned target does not exist: {...}'` |
| `RenameError` | `'File {...} drifted from the planned base; rebuild plan.'` |

#### `parse_frontmatter`

| Raises | Message |
|--------|---------|
| `SchemaError` | `'Malformed YAML in frontmatter: {...}'` |

## Source files

- `src/attune_rag/editor/__init__.py`
- `src/attune_rag/editor/schema.py`
- `src/attune_rag/editor/lint.py`
- `src/attune_rag/editor/autocomplete.py`
- `src/attune_rag/editor/references.py`
- `src/attune_rag/editor/rename.py`

## Tags

`editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
