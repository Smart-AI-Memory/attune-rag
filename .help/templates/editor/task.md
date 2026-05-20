---
type: task
name: editor-task
feature: editor
depth: task
generated_at: 2026-05-20T03:31:44.057284+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Work with the editor feature

Use the editor feature when you need to validate frontmatter, surface lint diagnostics, provide autocomplete suggestions, locate references, or rename aliases, tags, and template paths across a corpus.

## Prerequisites

- A configured `corpus` object that implements `CorpusProtocol`
- Python access to the `attune_rag.editor` package

## Validate and lint a template

Use these steps to check a template's frontmatter and surface any diagnostics before saving.

1. **Parse and validate the frontmatter.** Call `parse_frontmatter(yaml_text)` with the raw YAML block from the top of your template. The function returns a `(dict, list[FrontmatterIssue])` tuple. If the YAML is malformed, it raises `SchemaError`.

   ```python
   from attune_rag.editor import parse_frontmatter, SchemaError

   try:
       data, issues = parse_frontmatter(yaml_text)
   except SchemaError as e:
       print(f"Malformed YAML: {e}")
   ```

2. **Inspect schema violations.** Iterate `issues`. Each `FrontmatterIssue` has a `code`, `message`, and `path` tuple pointing to the offending key.

   ```python
   for issue in issues:
       print(issue.code, issue.path, issue.message)
   ```

3. **Run the full lint pass.** Call `lint_template(text, rel_path, corpus)` with the full template text, its corpus-relative path, and your corpus. Each returned `Diagnostic` carries a `severity`, `code`, `message`, and 1-indexed `line`/`col`/`end_line`/`end_col` range.

   ```python
   from attune_rag.editor import lint_template

   diagnostics = lint_template(text, "templates/my_template.md", corpus)
   for d in diagnostics:
       print(d.severity, d.code, d.line, d.message)
   ```

**Success criterion:** `parse_frontmatter` returns an empty `issues` list and `lint_template` returns an empty `diagnostics` list for a valid template.

## Provide autocomplete suggestions

Use these steps to populate tag or alias suggestions in an editor UI as the user types.

1. **Suggest tags.** Call `autocomplete_tags(corpus, prefix, limit=50)` with the current prefix string. The function returns up to `limit` matching tag strings.

   ```python
   from attune_rag.editor import autocomplete_tags

   suggestions = autocomplete_tags(corpus, prefix="data-")
   ```

2. **Suggest aliases.** Call `autocomplete_aliases(corpus, prefix, limit=50)` to get up to `limit` `AliasInfo` objects whose names start with `prefix`.

   ```python
   from attune_rag.editor import autocomplete_aliases

   suggestions = autocomplete_aliases(corpus, prefix="my_")
   ```

**Success criterion:** Both calls return a non-empty list when matching entries exist in the corpus, and an empty list when none match.

## Find references across a corpus

Use this step to locate every occurrence of an alias, tag, or template path before making a structural change.

1. **Call `find_references`.** Pass your corpus, the name to look up, and the appropriate `ReferenceKind` value.

   ```python
   from attune_rag.editor import find_references, ReferenceKind

   refs = find_references(corpus, "my_alias", ReferenceKind.ALIAS)
   for ref in refs:
       print(ref.template_path, ref.line, ref.col, ref.context)
   ```

   `find_references` raises `ValueError` if you pass an unsupported `ReferenceKind`.

**Success criterion:** The returned list contains a `Reference` entry for each file and line in the corpus that uses the specified name.

## Rename an alias, tag, or template path

Use these steps to rename a name across every file in the corpus atomically.

1. **Build the rename plan.** Call `plan_rename(corpus, old, new, kind)` to compute a `RenamePlan`. The plan contains `FileEdit` objects (with per-file `Hunk` diffs) and any `FileMove` objects for path renames. No files are modified at this step.

   ```python
   from attune_rag.editor import plan_rename, ReferenceKind

   plan = plan_rename(corpus, "old_alias", "new_alias", ReferenceKind.ALIAS)
   ```

   `plan_rename` raises `ValueError` for an unsupported `ReferenceKind`.

2. **Review the plan.** Inspect `plan.edits` and `plan.moves` before applying. Each `FileEdit` exposes `old_text`, `new_text`, and a list of `Hunk` objects you can display as a unified diff.

   ```python
   for edit in plan.edits:
       print(edit.path)
       for hunk in edit.hunks:
           print(hunk.header)
           print("".join(hunk.lines))
   ```

3. **Apply the plan.** Call `apply_rename(corpus, plan)` to write all edits and file moves to disk and refresh the corpus. The function returns a list of paths that were modified.

   ```python
   from attune_rag.editor import apply_rename

   modified = apply_rename(corpus, plan)
   print("Modified files:", modified)
   ```

   `apply_rename` raises `RenameCollisionError` if the proposed new name already exists, and `RenameError` for other failures such as a missing corpus root, a moved file that has drifted from the plan, or a missing source file.

**Success criterion:** `apply_rename` returns a list of modified paths with no exceptions, and calling `find_references(corpus, "old_alias", ReferenceKind.ALIAS)` afterwards returns an empty list.

## Key files

| File | Responsibility |
|---|---|
| `src/attune_rag/editor/__init__.py` | Public API surface |
| `src/attune_rag/editor/schema.py` | Frontmatter schema, `parse_frontmatter`, `validate_frontmatter`, `load_schema` |
| `src/attune_rag/editor/lint.py` | `lint_template` and `Diagnostic` |
| `src/attune_rag/editor/autocomplete.py` | `autocomplete_tags`, `autocomplete_aliases` |
| `src/attune_rag/editor/references.py` | `find_references` and `Reference` |
| `src/attune_rag/editor/rename.py` | `plan_rename`, `apply_rename`, `RenamePlan`, `FileEdit`, `FileMove`, `Hunk` |
