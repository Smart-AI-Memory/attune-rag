---
type: faq
name: editor-faq
feature: editor
depth: faq
generated_at: 2026-05-20T03:31:44.076531+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Editor FAQ

## What does the editor feature do?

It provides headless editor primitives for attune-rag templates: frontmatter parsing and validation, lint diagnostics, tag and alias autocomplete, cross-template reference lookup, and rename refactoring with atomic apply and rollback.

## What can I use it for?

Use it when you need any of the following against a template corpus:

- **Validate frontmatter** — `parse_frontmatter` parses a YAML block and returns structured `FrontmatterIssue` violations; `validate_frontmatter` checks already-parsed data against the JSON schema.
- **Lint a template** — `lint_template` runs all checks against a template's text and returns a list of `Diagnostic` objects with 1-indexed line and column positions.
- **Autocomplete** — `autocomplete_tags` and `autocomplete_aliases` prefix-match against the corpus and return up to `limit` suggestions.
- **Find references** — `find_references` locates every occurrence of an alias, tag, or template path across the corpus.
- **Rename** — `plan_rename` builds a `RenamePlan` (per-file `FileEdit` hunks plus any `FileMove` entries), and `apply_rename` writes the changes to disk and refreshes the corpus.

## Which function should I start with?

It depends on what you want to do:

| Goal | Function |
|---|---|
| Check a frontmatter block | `parse_frontmatter(yaml_text)` |
| Lint a full template | `lint_template(text, rel_path, corpus)` |
| Suggest tags while typing | `autocomplete_tags(corpus, prefix)` |
| Suggest aliases while typing | `autocomplete_aliases(corpus, prefix)` |
| Find where a name is used | `find_references(corpus, name, kind)` |
| Preview a rename | `plan_rename(corpus, old, new, kind)` |
| Apply a rename to disk | `apply_rename(corpus, plan)` |

## What is a RenamePlan and why is there a separate apply step?

`plan_rename` computes all the edits without touching disk — it returns a `RenamePlan` that lists `FileEdit` hunks (old and new text per file) and `FileMove` entries (path changes). You can inspect or serialize the plan with `to_dict()` before committing. Call `apply_rename` when you're ready to write the changes and refresh the corpus.

## What errors can apply_rename raise?

`apply_rename` raises a `RenameError` (or its subclass `RenameCollisionError`) in these situations:

- The corpus has no resolvable root path.
- A source file is missing at apply time.
- The proposed new name already exists in the corpus (`RenameCollisionError`).
- A file move fails at the OS level.
- A planned target file doesn't exist after the move.
- A file's content drifted from the plan's baseline — in this case, rebuild the plan before retrying.

## What does RenameCollisionError tell me?

It carries the conflicting `name` and the `owning_path` of the file that already uses that name, so you can show the user exactly what's in the way.

## When does parse_frontmatter raise SchemaError instead of returning issues?

`parse_frontmatter` raises `SchemaError` only when the YAML itself is malformed and cannot be parsed at all. Structural problems that are valid YAML — missing required fields, wrong value types, and so on — come back as `FrontmatterIssue` entries in the returned list, not as exceptions.

## Are line and column numbers 0-indexed or 1-indexed?

`Diagnostic` line and column fields (`line`, `col`, `end_line`, `end_col`) are **1-indexed**. `FrontmatterIssue` line and column values are also 1-indexed within the frontmatter block.

## How do I debug unexpected results?

Run the feature's tests first:

```
pytest -k "editor" -v
```

If the tests pass but your code still misbehaves, add a `logger.debug` call at the point where results diverge from what you expect, then re-run with debug logging enabled.

## Where are the source files?

| File | Responsibility |
|---|---|
| `src/attune_rag/editor/__init__.py` | Public API surface |
| `src/attune_rag/editor/schema.py` | `load_schema`, `parse_frontmatter`, `validate_frontmatter` |
| `src/attune_rag/editor/lint.py` | `lint_template`, `Diagnostic`, `Severity` |
| `src/attune_rag/editor/autocomplete.py` | `autocomplete_tags`, `autocomplete_aliases` |
| `src/attune_rag/editor/references.py` | `find_references`, `Reference`, `ReferenceKind` |
| `src/attune_rag/editor/rename.py` | `plan_rename`, `apply_rename`, `RenamePlan`, `FileEdit`, `FileMove` |

**Tags:** `editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
