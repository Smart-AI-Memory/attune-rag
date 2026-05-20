---
type: task
name: editor-task
feature: editor
depth: task
generated_at: 2026-05-20T02:44:35.485551+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Work with the template editor

Use the editor module when you need to lint templates, resolve cross-references, provide autocomplete suggestions, or rename aliases and tags across a corpus.

## Prerequisites

- Access to the project source code
- A corpus object that implements `CorpusProtocol`, available to pass into editor functions

## Steps

1. **Load and validate the frontmatter schema.**
   Call `load_schema()` to retrieve the parsed JSON Schema that defines the v1 frontmatter contract. Then call `parse_frontmatter(yaml_text)` to parse a raw YAML frontmatter block and receive both the parsed data and any `FrontmatterIssue` violations in a single call. If the YAML is malformed, `parse_frontmatter` raises `SchemaError`.

   ```python
   from attune_rag.editor import load_schema, parse_frontmatter

   schema = load_schema()
   data, issues = parse_frontmatter(yaml_text)
   ```

   If you already have parsed frontmatter data, call `validate_frontmatter(data)` directly to get a list of `FrontmatterIssue` objects without re-parsing.

2. **Lint a template.**
   Call `lint_template(text, rel_path, corpus)` to run all lint checks against a template's full text. The function returns a list of `Diagnostic` objects. Each `Diagnostic` carries a `severity`, `code`, `message`, and 1-indexed `line`, `col`, `end_line`, and `end_col` fields.

   ```python
   from attune_rag.editor import lint_template

   diagnostics = lint_template(text, rel_path="tasks/my-task.md", corpus=corpus)
   for d in diagnostics:
       print(d.severity, d.code, d.message, f"{d.line}:{d.col}")
   ```

3. **Provide autocomplete suggestions.**
   Call `autocomplete_tags(corpus, prefix, limit)` to get up to `limit` tag suggestions starting with `prefix`. Call `autocomplete_aliases(corpus, prefix, limit)` to get up to `limit` `AliasInfo` suggestions instead.

   ```python
   from attune_rag.editor import autocomplete_tags, autocomplete_aliases

   tags = autocomplete_tags(corpus, prefix="migr", limit=10)
   aliases = autocomplete_aliases(corpus, prefix="con-", limit=10)
   ```

4. **Find references to a name.**
   Call `find_references(corpus, name, kind)` to locate every occurrence of an alias, tag, or template path across the corpus. The function returns a list of `Reference` objects, each with a `template_path`, `line`, `col`, and `context`. Pass a `ReferenceKind` value for `kind`; the function raises `ValueError` for unsupported kinds.

   ```python
   from attune_rag.editor import find_references, ReferenceKind

   refs = find_references(corpus, name="con-error-handling", kind=ReferenceKind.ALIAS)
   for ref in refs:
       print(ref.template_path, ref.line, ref.col)
   ```

5. **Plan a rename.**
   Call `plan_rename(corpus, old, new, kind)` to compute a `RenamePlan` before touching any files. The plan contains `FileEdit` objects (with per-file diff hunks) and `FileMove` objects for any path changes. Review the plan before applying it.

   ```python
   from attune_rag.editor import plan_rename, ReferenceKind

   plan = plan_rename(corpus, old="con-error-handling", new="con-error-design", kind=ReferenceKind.ALIAS)
   for edit in plan.edits:
       print(edit.path, edit.old_text, "->", edit.new_text)
   ```

6. **Apply the rename.**
   Call `apply_rename(corpus, plan)` to write the planned edits to disk and refresh the corpus atomically. The function raises `RenameCollisionError` if the new name already exists, and raises `RenameError` for missing files, drift between plan and disk state, or a corpus without a resolvable root path.

   ```python
   from attune_rag.editor import apply_rename

   changed_paths = apply_rename(corpus, plan)
   ```

## Key files

| File | Responsibility |
|---|---|
| `src/attune_rag/editor/__init__.py` | Public API surface |
| `src/attune_rag/editor/schema.py` | `load_schema`, `validate_frontmatter`, `parse_frontmatter` |
| `src/attune_rag/editor/lint.py` | `lint_template`, `Diagnostic`, `Severity` |
| `src/attune_rag/editor/autocomplete.py` | `autocomplete_tags`, `autocomplete_aliases` |
| `src/attune_rag/editor/references.py` | `find_references`, `Reference`, `ReferenceKind` |
| `src/attune_rag/editor/rename.py` | `plan_rename`, `apply_rename`, `RenamePlan`, `FileEdit`, `FileMove`, `Hunk` |

## Verify the task worked

Run the editor test suite to confirm all functions behave correctly:

```bash
pytest -k "editor"
```

All tests pass with no errors or failures. If `lint_template` returns diagnostics on a known-valid template, or `apply_rename` raises `RenameError`, re-check that the corpus object is fully initialized and that the frontmatter schema matches the template under test.
