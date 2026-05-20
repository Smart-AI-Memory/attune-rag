---
type: quickstart
name: editor-quickstart
feature: editor
depth: quickstart
generated_at: 2026-05-20T03:31:44.079009+00:00
source_hash: 1781a70216d482b69e33e146fcc3a1f37451550a76ce813bd81d0e3694790e4a
status: generated
---

# Quickstart: attune-rag editor

Lint a template file and see its diagnostics in under a minute:

```python
from attune_rag.editor import lint_template

text = open("templates/my_template.yaml").read()
diagnostics = lint_template(text, rel_path="templates/my_template.yaml", corpus=my_corpus)

for d in diagnostics:
    print(f"[{d.severity.name}] {d.code} line {d.line}:{d.col} — {d.message}")
```

Expected output (no issues):

```
# (empty — no diagnostics means the template is clean)
```

Expected output (with issues):

```
[ERROR] E101 line 4:3 — Missing required frontmatter field: 'alias'
[WARNING] W201 line 9:1 — Tag 'draft' is not defined in the corpus
```

## Prerequisites

- attune-rag is installed in your Python environment (`pip install attune-rag`)
- You have a `corpus` object that implements `CorpusProtocol`

## Steps

1. **Parse and validate frontmatter.** Before linting a full template, check that its YAML frontmatter is well-formed:

    ```python
    from attune_rag.editor import parse_frontmatter

    yaml_block = """
    alias: my-template
    tags: [draft, review]
    """

    data, issues = parse_frontmatter(yaml_block)
    for issue in issues:
        print(f"{issue.code}: {issue.message} (at {issue.path})")
    ```

    `parse_frontmatter` raises `SchemaError` if the YAML is malformed. `issues` is a list of `FrontmatterIssue` objects for schema violations — an empty list means the frontmatter is valid.

2. **Lint a full template.** Call `lint_template` with the file's text, its corpus-relative path, and the corpus. It returns a list of `Diagnostic` objects with 1-indexed line and column positions:

    ```python
    from attune_rag.editor import lint_template

    diagnostics = lint_template(text, rel_path="templates/my_template.yaml", corpus=my_corpus)
    for d in diagnostics:
        print(d.to_dict())
    ```

3. **Find references and plan a rename.** Once the template is clean, you can locate all uses of a name and rename them across the corpus:

    ```python
    from attune_rag.editor import find_references, plan_rename, apply_rename, ReferenceKind

    refs = find_references(my_corpus, name="old-alias", kind=ReferenceKind.ALIAS)
    print(f"Found {len(refs)} reference(s)")

    plan = plan_rename(my_corpus, old="old-alias", new="new-alias", kind=ReferenceKind.ALIAS)
    changed_paths = apply_rename(my_corpus, plan)
    print("Updated files:", changed_paths)
    ```

    `apply_rename` writes changes to disk and refreshes the corpus. It raises `RenameCollisionError` if `new-alias` already exists.

## Next

Explore autocomplete for the template editor — call `autocomplete_tags(corpus, prefix="dr")` or `autocomplete_aliases(corpus, prefix="my")` to see prefix-matched suggestions returned as lists you can feed directly into an editor UI.
