---
type: concept
name: editor-concept
feature: editor
depth: concept
generated_at: 2026-05-20T02:44:35.480495+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Editor

The attune-rag editor is a suite of headless, pure-function primitives that give any host — the `attune-gui` editor or the `attune-author` CLI — schema validation, live linting, autocomplete, reference lookup, and rename refactoring for template files.

## What the editor does

The editor exposes six distinct capabilities, each implemented as a standalone function that operates over a `CorpusProtocol` value:

| Capability | Entry point | What it produces |
|---|---|---|
| **Schema loading** | `load_schema()` | The parsed JSON Schema for template frontmatter |
| **Frontmatter parsing** | `parse_frontmatter(yaml_text)` | A validated dict plus any `FrontmatterIssue` violations |
| **Linting** | `lint_template(text, rel_path, corpus)` | A list of 1-indexed `Diagnostic` objects |
| **Autocomplete** | `autocomplete_tags(corpus, prefix)` / `autocomplete_aliases(corpus, prefix)` | Up to 50 prefix-matched tag or alias suggestions |
| **Reference lookup** | `find_references(corpus, name, kind)` | Every occurrence of an alias, tag, or template path across the corpus |
| **Rename refactoring** | `plan_rename(…)` → `apply_rename(…)` | A `RenamePlan` of `FileEdit` hunks and `FileMove` records, applied atomically with rollback |

Because every function accepts a corpus and returns plain data, the editor has no UI dependencies — the same code runs identically in the GUI and the CLI.

## How the pieces fit together

A typical editing session flows through the capabilities in layers:

1. **Schema** — `load_schema` and `parse_frontmatter` establish whether a template's YAML frontmatter is structurally valid before any other check runs. If the YAML is malformed, `parse_frontmatter` raises `SchemaError` immediately.

2. **Lint** — `lint_template` runs all checks against the full template text and returns `Diagnostic` values with 1-indexed line and column positions that a host can map directly to editor gutter markers.

3. **Autocomplete** — As the author types a tag or alias, `autocomplete_tags` and `autocomplete_aliases` prefix-match against the live corpus so suggestions always reflect what actually exists.

4. **References and rename** — Before renaming a tag, alias, or template path, `find_references` shows every location that will be affected. `plan_rename` then computes a `RenamePlan` — a list of `FileEdit` hunks (in unified-diff format) and `FileMove` records — which `apply_rename` executes atomically. If any step fails (missing file, target collision, content drift), `apply_rename` raises a `RenameError` subclass and leaves the corpus unchanged.

## Key data types

| Type | Role |
|---|---|
| `Diagnostic` | A single lint finding: severity, code, message, and 1-indexed start/end position |
| `FrontmatterIssue` | A single schema violation discovered during frontmatter parsing |
| `Reference` | One occurrence of a name in the corpus: template path, line, column, and context |
| `Hunk` | A single unified-diff hunk produced by `plan_rename` |
| `FileEdit` | All hunks for one file: old text, new text, and the hunk list |
| `FileMove` | A planned path rename for a template file |
| `RenamePlan` | The complete rename operation: old name, new name, kind, edits, and moves |

## When the editor matters

You interact with editor primitives whenever you need to:

- Validate that a template's frontmatter conforms to the `template_schema.json` contract before committing it.
- Surface lint diagnostics in a text editor without coupling the lint logic to any particular UI toolkit.
- Populate an autocomplete dropdown with tags or aliases that exist in the current corpus.
- Safely rename a tag, alias, or template across hundreds of files and have the change applied as a single atomic operation.
