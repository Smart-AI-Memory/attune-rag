---
type: concept
name: editor-concept
feature: editor
depth: concept
generated_at: 2026-05-20T03:31:44.051205+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Editor

## Overview

The `editor` feature is a collection of headless, pure-function primitives that give attune-rag IDEs and CLI tools schema validation, linting, autocomplete, reference lookup, and cross-template rename refactoring — all operating against a `CorpusProtocol` without coupling to any specific UI layer.

The five capabilities form a pipeline that mirrors what a language server does for source code:

1. **Schema validation** — `load_schema` loads the JSON Schema for template frontmatter. `parse_frontmatter` parses a raw YAML block and validates it in one step, raising `SchemaError` on malformed YAML and returning a list of `FrontmatterIssue` objects for any schema violations. `validate_frontmatter` handles the validation step alone when the YAML has already been parsed.

2. **Linting** — `lint_template` runs all checks against a template's text and returns a list of `Diagnostic` objects. Each `Diagnostic` carries a `severity`, an error `code`, a human-readable `message`, and 1-indexed `line`/`col`/`end_line`/`end_col` span fields.

3. **Autocomplete** — `autocomplete_tags` and `autocomplete_aliases` prefix-match against the corpus and return up to `limit` suggestions (default 50). Tags are returned as plain strings; aliases are returned as `AliasInfo` objects.

4. **Reference lookup** — `find_references` searches every file in the corpus for occurrences of a given alias, tag, or template path (selected by `ReferenceKind`) and returns a list of `Reference` objects. Each `Reference` records the `template_path`, `line`, `col`, and a `ReferenceContext` describing how the name is used.

5. **Rename refactoring** — `plan_rename` computes a `RenamePlan` that collects all necessary `FileEdit` and `FileMove` operations without touching disk. Each `FileEdit` breaks its changes into `Hunk` objects in unified-diff format so callers can preview the diff before committing. `apply_rename` then executes the plan atomically, refreshes the corpus, and raises a `RenameCollisionError` if the proposed new name already exists, or a `RenameError` subclass for problems such as a missing move source or a file that has drifted from the planned base.

## Integration points

Other parts of the codebase consume the `editor` feature through these public interfaces:

| Interface | Role | Source file |
|-----------|------|-------------|
| `Diagnostic` | Lint result with location and severity | `src/attune_rag/editor/lint.py` |
| `Reference` | Single corpus-wide reference to a name | `src/attune_rag/editor/references.py` |
| `RenamePlan` | Diff preview before writing to disk | `src/attune_rag/editor/rename.py` |
| `RenameError` / `RenameCollisionError` | Failure signals from `apply_rename` | `src/attune_rag/editor/rename.py` |
| `Hunk` | Unified-diff fragment within a `FileEdit` | `src/attune_rag/editor/rename.py` |

The attune-gui editor and the `attune-author edit` CLI are the primary consumers. Because every function accepts a `CorpusProtocol` value rather than a concrete corpus type, any component that satisfies the protocol can use the full feature set without additional glue code.
