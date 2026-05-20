---
type: comparison
name: editor-comparison
feature: editor
depth: comparison
generated_at: 2026-05-20T03:31:44.086252+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Comparison: Editor primitives vs alternatives

## What the editor module provides

The `editor` module is a suite of headless, pure-function primitives built on top of `CorpusProtocol`. It covers five distinct capabilities, each with a dedicated sub-module:

| Capability | Entry point(s) | What it does |
|---|---|---|
| **Schema validation** | `load_schema()`, `parse_frontmatter()`, `validate_frontmatter()` | Parses the v1 frontmatter JSON Schema; splits a YAML block and returns typed `FrontmatterIssue` violations |
| **Linting** | `lint_template()` | Runs all lint checks against template text and returns 1-indexed `Diagnostic` objects with severity, code, message, and span |
| **Autocomplete** | `autocomplete_tags()`, `autocomplete_aliases()` | Prefix-matches tags or aliases against the corpus; returns up to `limit` results (default 50) |
| **Reference lookup** | `find_references()` | Locates every occurrence of an alias, tag, or template path across the full corpus |
| **Rename refactor** | `plan_rename()`, `apply_rename()` | Builds a `RenamePlan` with per-file `FileEdit` hunks and `FileMove` records; applies the plan atomically with rollback on failure |

These functions are consumed directly by `attune-gui` (live editor) and the `attune-author edit` CLI.

## Capability tradeoffs

### Schema validation: `parse_frontmatter` vs `validate_frontmatter`

You have two paths for checking frontmatter:

| | `parse_frontmatter(yaml_text)` | `validate_frontmatter(data)` |
|---|---|---|
| **Input** | Raw YAML string | Already-parsed `dict` |
| **Raises** | `SchemaError` on malformed YAML | Never raises — violations are returned |
| **Returns** | `(dict, list[FrontmatterIssue])` | `list[FrontmatterIssue]` |
| **Best for** | Reading directly from a file buffer | Validating data you already parsed elsewhere |

Use `parse_frontmatter` when you start from raw text. Use `validate_frontmatter` when your pipeline has already parsed the YAML and you only need the schema check.

### Rename: `plan_rename` vs `apply_rename`

The rename refactor is intentionally split into two steps:

| | `plan_rename()` | `apply_rename()` |
|---|---|---|
| **Side effects** | None — pure computation | Writes to disk and refreshes the corpus |
| **Returns** | `RenamePlan` (edits + moves as diffs) | `list[str]` of affected paths |
| **Can be previewed** | Yes — inspect `RenamePlan.edits` and `RenamePlan.moves` | No — changes are applied immediately |
| **Raises** | `ValueError` for unsupported `ReferenceKind` | `RenameCollisionError` if the new name already exists; `RenameError` for missing files, drift, or no resolvable root |

Always call `plan_rename` first and inspect the `RenamePlan` before calling `apply_rename`. The plan is invalidated if template files change between the two calls — `apply_rename` will raise `RenameError` with a "drifted from the planned base" message in that case.

### Lint vs schema validation

These two capabilities look similar but cover different layers:

| | `lint_template()` | `parse_frontmatter()` / `validate_frontmatter()` |
|---|---|---|
| **Scope** | Full template text (body + frontmatter) | Frontmatter only |
| **Output type** | `Diagnostic` (severity, code, line, col, span) | `FrontmatterIssue` (code, message, path tuple) |
| **Corpus-aware** | Yes — checks references against the corpus | No — purely structural |
| **Use case** | Live editor diagnostics | CI schema enforcement, import validation |

## When to use the editor module

**Use `editor` when you are building or extending editor tooling** — an IDE plugin, a language server, a CLI author workflow, or a CI check that needs structured diagnostics. Specifically:

- You need **live autocomplete** for tags or aliases as the user types → `autocomplete_tags()` / `autocomplete_aliases()`
- You need **structured lint results** with line/column spans to render in an editor → `lint_template()`
- You need to **find all references** to an alias, tag, or template path before refactoring → `find_references()`
- You need a **safe, previewable rename** across the corpus → `plan_rename()` then `apply_rename()`
- You need to **validate or enforce** the frontmatter schema in CI or on import → `parse_frontmatter()` / `validate_frontmatter()`

## When to use something else

- **You only need to query or retrieve templates** (no authoring, no diagnostics): use the corpus retrieval layer directly rather than routing through editor primitives.
- **You need to orchestrate multiple features together** (for example, lint then rename then re-index): use the higher-level orchestration layer above `editor` rather than chaining these functions yourself.
- **You need behavior the public API does not expose**: do not patch internals. The `__all__` export list in each sub-module is the stable surface — anything outside it is subject to change without notice.
- **You are writing a one-off exploration script**: wiring up `plan_rename` and `apply_rename` correctly (including drift detection) is non-trivial. A read-only script against the corpus is a simpler starting point.

## Source files

- `src/attune_rag/editor/__init__.py`
- `src/attune_rag/editor/schema.py`
- `src/attune_rag/editor/lint.py`
- `src/attune_rag/editor/autocomplete.py`
- `src/attune_rag/editor/references.py`
- `src/attune_rag/editor/rename.py`

**Tags:** `editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
