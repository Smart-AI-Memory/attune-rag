---
type: note
name: editor-note
feature: editor
depth: note
generated_at: 2026-05-20T03:31:44.083713+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Note: editor

## Overview

The `attune_rag.editor` package provides headless editor-support primitives for attune-rag templates. It covers six concerns, each in its own source file:

| File | Responsibility |
|---|---|
| `schema.py` | Load and validate the JSON Schema for template frontmatter (`template_schema.json`) |
| `lint.py` | Run lint checks and return `Diagnostic` objects with 1-indexed line/column positions |
| `autocomplete.py` | Prefix-match tags and aliases against a corpus |
| `references.py` | Find every occurrence of an alias, tag, or template path across a corpus |
| `rename.py` | Build and apply a `RenamePlan` for cross-template rename refactors |
| `__init__.py` | Re-export the full public surface listed in `__all__` |

The package is consumed by both the `attune-gui` editor (live feedback) and the `attune-author` edit CLI.

## Design

All top-level functions (`lint_template`, `find_references`, `plan_rename`, `apply_rename`, and the autocomplete functions) are pure over a `CorpusProtocol` argument — they do not hold state themselves. The data classes (`Diagnostic`, `Reference`, `Hunk`, `FileEdit`, `FileMove`, `RenamePlan`) act as typed return values and each expose a `to_dict()` method for serialization.

The rename workflow is two-phase by design. `plan_rename` computes a `RenamePlan` — a list of `FileEdit` hunks and `FileMove` records — without touching disk. `apply_rename` then executes that plan atomically and refreshes the corpus. Separating planning from application lets callers preview or reject changes before committing them. `apply_rename` guards against several failure modes at apply time: a missing corpus root, a missing move source, a name collision (`RenameCollisionError`), and file drift since the plan was built.

Frontmatter handling is also split into two stages. `parse_frontmatter` parses raw YAML and raises `SchemaError` on malformed input. `validate_frontmatter` checks already-parsed data against the JSON Schema and returns a list of `FrontmatterIssue` objects; it does not raise on schema violations, which lets callers collect all issues in one pass rather than stopping at the first error.

**Tags:** `editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
