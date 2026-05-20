---
type: troubleshooting
name: editor-troubleshooting
feature: editor
depth: troubleshooting
generated_at: 2026-05-20T03:31:44.073971+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Troubleshoot editor

## Before you start

The `editor` module provides headless editing primitives for attune-rag templates. It covers frontmatter parsing and schema validation (`parse_frontmatter`, `validate_frontmatter`, `load_schema`), lint diagnostics (`lint_template`), prefix-match autocomplete for tags and aliases, cross-corpus reference lookup (`find_references`), and rename refactoring (`plan_rename`, `apply_rename`). These are pure functions over a `CorpusProtocol` object. The same functions back both the `attune-gui` editor and the `attune-author` edit CLI.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `SchemaError: Malformed YAML in frontmatter` | The raw YAML block passed to `parse_frontmatter()` — look for tabs, unquoted colons, or invalid Unicode |
| `validate_frontmatter()` returns unexpected `FrontmatterIssue` entries | Compare your frontmatter keys and types against the schema returned by `load_schema()` — `FrontmatterIssue.path` points to the offending key |
| `lint_template()` returns no diagnostics when errors are expected | Confirm you are passing the correct `rel_path` and a live corpus object — a mismatched path silently skips cross-template checks |
| `find_references()` raises `ValueError: Unsupported reference kind` | Verify the `kind` argument is a valid `ReferenceKind` enum member, not a raw string |
| `plan_rename()` raises `ValueError: Unsupported rename kind` | Same as above — check that `kind` is a `ReferenceKind` value |
| `apply_rename()` raises `RenameCollisionError` | The proposed new name already exists in the corpus; check `RenameCollisionError.name` and `owning_path` to identify the conflict |
| `apply_rename()` raises `RenameError: Corpus has no resolvable root path` | The corpus passed to `apply_rename()` does not expose a filesystem root — this operation requires a corpus backed by a real directory |
| `apply_rename()` raises `RenameError: File ... drifted from the planned base` | The file was modified on disk between calling `plan_rename()` and `apply_rename()` — rebuild the plan |
| Autocomplete returns an empty list | Confirm the `prefix` string matches the case used in the corpus and that `limit` is greater than zero |

## Step-by-step diagnosis

Work from cheapest to most expensive.

1. **Reproduce the failure in isolation.**
   Call the failing function directly with the minimal required arguments. For example, to isolate a lint issue:

   ```python
   from attune_rag.editor.lint import lint_template
   diagnostics = lint_template(text="<your template>", rel_path="test.md", corpus=corpus)
   print([d.to_dict() for d in diagnostics])
   ```

   Confirm the failure occurs without any surrounding application code.

2. **Inspect return values and raised exceptions.**
   All public functions return typed results or raise documented exceptions. Check:
   - `Diagnostic.severity`, `code`, `message`, `line`, and `col` for lint failures.
   - `FrontmatterIssue.code`, `message`, and `path` for schema violations.
   - `Reference.template_path`, `line`, and `context` for unexpected reference results.
   - The exception message text for `RenameError` and its subclasses — each message names the specific file or value involved.

3. **Validate your corpus object.**
   Most functions accept `corpus: Any` typed against `CorpusProtocol`. If results are silently wrong or empty, confirm the corpus is fully loaded and up to date. A stale or partially initialised corpus is a common source of empty autocomplete results and missing references.

4. **Check the `RenamePlan` before applying it.**
   Call `plan_rename()` and inspect the result with `plan.to_dict()` before passing it to `apply_rename()`. Verify `plan.edits` contains the expected `FileEdit` hunks and `plan.moves` contains the expected `FileMove` entries. This lets you catch collision or path issues without touching the filesystem.

5. **Run the editor tests.**
   ```bash
   pytest -k "editor" -v
   ```
   If a test covers the failing path, use its fixtures to reproduce the issue. A failing test here confirms a regression; a passing test with a failing manual call usually points to a corpus or input mismatch in your calling code.

## Common fixes

- **Malformed frontmatter YAML.** Fix tabs-instead-of-spaces, unquoted special characters, or duplicate keys in the frontmatter block, then re-run `parse_frontmatter()`. The `SchemaError` message includes the YAML parser's description of the problem.

- **Wrong `ReferenceKind` value.** Both `find_references()` and `plan_rename()` reject an unsupported `kind` with a `ValueError`. Use the enum directly:
  ```python
  from attune_rag.editor.references import ReferenceKind
  refs = find_references(corpus, name="my-alias", kind=ReferenceKind.ALIAS)
  ```

- **Rename collision.** If `apply_rename()` raises `RenameCollisionError`, either choose a different target name or remove/rename the existing item at `owning_path` first. The error exposes both `name` and `owning_path` for inspection.

- **Drifted file during rename.** If you see `RenameError: File ... drifted from the planned base`, the template was edited between planning and applying. Rebuild the plan:
  ```python
  plan = plan_rename(corpus, old="old-name", new="new-name", kind=ReferenceKind.ALIAS)
  apply_rename(corpus, plan)
  ```

- **Corpus has no root path.** `apply_rename()` requires a corpus with a resolvable filesystem root. If you are working with an in-memory corpus (for example, in tests), you cannot call `apply_rename()` — use `plan_rename()` to inspect the intended edits without applying them.

- **Empty autocomplete results.** If `autocomplete_tags()` or `autocomplete_aliases()` return fewer results than expected, check that `prefix` is correct (these are prefix matches, not substring matches) and that your `limit` argument is large enough. The default `limit` is `50`.

## Source files

- `src/attune_rag/editor/__init__.py`
- `src/attune_rag/editor/schema.py`
- `src/attune_rag/editor/lint.py`
- `src/attune_rag/editor/autocomplete.py`
- `src/attune_rag/editor/references.py`
- `src/attune_rag/editor/rename.py`


**Tags:** `editor`, `lint`, `rename`, `autocomplete`, `schema`, `references`, `refactor`, `template`
