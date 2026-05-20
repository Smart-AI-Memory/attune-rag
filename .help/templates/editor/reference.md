---
type: reference
name: editor-reference
feature: editor
depth: reference
generated_at: 2026-05-20T03:31:44.061780+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Editor reference

Headless primitives for building template-editor tooling. Use these functions and classes to validate frontmatter, lint template text, prefix-match tags and aliases against a corpus, locate all references to an alias, tag, or path, and plan or apply cross-template renames with per-file diff hunks and atomic rollback. All functions accept a `CorpusProtocol` value and are used directly by the attune-gui editor and the `attune-author edit` CLI.

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
| `RenamePlan` | A complete rename plan, including all file edits and moves. |
| `SchemaError` | Raised when the frontmatter cannot be parsed at all (malformed YAML). |
| `FrontmatterIssue` | A single schema violation. Line and column are 1-indexed within the frontmatter block. |

### `Diagnostic`

`[dataclass]` — `src/attune_rag/editor/lint.py`

A single lint diagnostic.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `severity` | `Severity` | |
| `code` | `str` | |
| `message` | `str` | |
| `line` | `int` | |
| `col` | `int` | |
| `end_line` | `int` | |
| `end_col` | `int` | |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serialize the diagnostic to a dictionary. |

---

### `Reference`

`[dataclass]` — `src/attune_rag/editor/references.py`

A single reference to a name in a corpus.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `template_path` | `str` | |
| `line` | `int` | |
| `col` | `int` | |
| `context` | `ReferenceContext` | |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serialize the reference to a dictionary. |

---

### `RenameError`

`src/attune_rag/editor/rename.py`

Base class for rename refactor failures.

---

### `RenameCollisionError`

`src/attune_rag/editor/rename.py`

Raised when the proposed new name already exists.

#### Constructor

| Parameters | Returns |
|------------|---------|
| `name: str, owning_path: str` | `None` |

---

### `Hunk`

`[dataclass]` — `src/attune_rag/editor/rename.py`

A single unified-diff hunk.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `hunk_id` | `str` | |
| `header` | `str` | |
| `lines` | `list[str]` | `field(default_factory=list)` |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serialize the hunk to a dictionary. |

---

### `FileEdit`

`[dataclass]` — `src/attune_rag/editor/rename.py`

A planned edit to a single template file.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `path` | `str` | |
| `old_text` | `str` | |
| `new_text` | `str` | |
| `hunks` | `list[Hunk]` | `field(default_factory=list)` |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serialize the file edit to a dictionary. |

---

### `FileMove`

`[dataclass]` — `src/attune_rag/editor/rename.py`

A planned file move (rename of a template's rel-path).

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `old_path` | `str` | |
| `new_path` | `str` | |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serialize the file move to a dictionary. |

---

### `RenamePlan`

`[dataclass]` — `src/attune_rag/editor/rename.py`

A complete rename plan, including all file edits and moves.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `old` | `str` | |
| `new` | `str` | |
| `kind` | `ReferenceKind` | |
| `edits` | `list[FileEdit]` | `field(default_factory=list)` |
| `moves` | `list[FileMove]` | `field(default_factory=list)` |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `to_dict` | `self` | `dict[str, Any]` | Serialize the rename plan to a dictionary. |

---

### `SchemaError`

`src/attune_rag/editor/schema.py`

Raised when the frontmatter cannot be parsed at all (malformed YAML).

---

### `FrontmatterIssue`

`[dataclass]` — `src/attune_rag/editor/schema.py`

A single schema violation. Line and column are 1-indexed within the frontmatter block.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `code` | `str` | |
| `message` | `str` | |
| `path` | `tuple[str \| int, ...]` | `()` |

---

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

### `find_references`

`src/attune_rag/editor/references.py`

#### Raises

| Exception | Message |
|-----------|---------|
| `ValueError` | `'Unsupported reference kind: {...}'` |

---

### `plan_rename`

`src/attune_rag/editor/rename.py`

#### Raises

| Exception | Message |
|-----------|---------|
| `ValueError` | `'Unsupported rename kind: {...}'` |

---

### `apply_rename`

`src/attune_rag/editor/rename.py`

#### Raises

| Exception | Message |
|-----------|---------|
| `RenameError` | `'Corpus has no resolvable root path; apply is not supported.'` |
| `RenameError` | `'Move source missing at apply time: {...}'` |
| `RenameCollisionError` | |
| `RenameError` | `'Failed to move {...} -> {...}: {...}'` |
| `RenameError` | `'Planned target does not exist: {...}'` |
| `RenameError` | `'File {...} drifted from the planned base; rebuild plan.'` |

---

### `parse_frontmatter`

`src/attune_rag/editor/schema.py`

#### Raises

| Exception | Message |
|-----------|---------|
| `SchemaError` | `'Malformed YAML in frontmatter: {...}'` |

---

## Constants

| Constant | Type | Members |
|----------|------|---------|
| `_PATH_KEYED_SIDECARS` | `tuple` | `'summaries.json'`, `'summaries_by_path.json'` |
| `_SCHEMA_RESOURCE` | `str` | `'template_schema.json'` |

## Source files

- `src/attune_rag/editor/__init__.py`
- `src/attune_rag/editor/schema.py`
- `src/attune_rag/editor/lint.py`
- `src/attune_rag/editor/autocomplete.py`
- `src/attune_rag/editor/references.py`
- `src/attune_rag/editor/rename.py`

## Tags

`editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
