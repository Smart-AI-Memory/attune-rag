---
type: task
name: corpus-task
feature: corpus
depth: task
generated_at: 2026-05-20T03:23:24.651571+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Work with corpus

Use a corpus implementation when you need to load and retrieve help templates — choose `DirectoryCorpus` to read markdown files from disk, `AttuneHelpCorpus` to wrap the bundled attune-help templates, or implement `CorpusProtocol` directly to supply your own source.

## Prerequisites

- Access to the project source code
- Python `Path` available if you are instantiating `DirectoryCorpus`

## Key files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py` — `CorpusProtocol`, `RetrievalEntry`, `AliasInfo`, `DuplicateAliasError`
- `src/attune_rag/corpus/directory.py` — `DirectoryCorpus`
- `src/attune_rag/corpus/attune_help.py` — `AttuneHelpCorpus`

## Steps

1. **Choose the right corpus class for your use case.**

   | Goal | Class to use |
   |---|---|
   | Load markdown files from a local directory | `DirectoryCorpus` |
   | Wrap the bundled attune-help templates | `AttuneHelpCorpus` |
   | Supply a custom template source | Implement `CorpusProtocol` |

2. **Instantiate the corpus.**

   - **`DirectoryCorpus`** — point it at a directory of `.md` files:
     ```python
     from pathlib import Path
     from attune_rag.corpus import DirectoryCorpus

     corpus = DirectoryCorpus(
         root=Path("path/to/templates"),
         summaries_file="summaries.json",   # optional
         cross_links_file="links.json",     # optional
         cache=True,                        # cache entries after first load
     )
     ```
     By default it globs `**/*.md`. Pass a custom `glob` string to restrict or expand which files are loaded.

   - **`AttuneHelpCorpus`** — use the class method to load the bundled templates:
     ```python
     from attune_rag.corpus import AttuneHelpCorpus

     corpus = AttuneHelpCorpus.from_attune_help()
     ```

3. **Retrieve entries from the corpus.**

   - Iterate over every entry:
     ```python
     for entry in corpus.entries():
         print(entry.path, entry.category, entry.summary)
     ```
   - Fetch a single entry by path:
     ```python
     entry = corpus.get("errors/not-found.md")
     if entry is None:
         print("Template not found.")
     ```
   Each `RetrievalEntry` exposes `path`, `category`, `content`, and optionally `summary`, `related`, `aliases`, and `metadata`.

4. **Inspect the alias and path indexes (DirectoryCorpus only).**

   Use `path_index` for a `{rel_path: RetrievalEntry}` mapping and `alias_index` for a `{alias: AliasInfo}` mapping:
   ```python
   entry = corpus.path_index.get("how-to/configure.md")
   alias_info = corpus.alias_index.get("configure")
   ```
   If two templates declare the same alias, `DirectoryCorpus` raises `DuplicateAliasError` with the conflicting paths. Catch it explicitly if you load untrusted template directories.

5. **Implement a custom corpus (optional).**

   Create a class that satisfies `CorpusProtocol` by implementing `entries()`, `get()`, `name`, and `version`:
   ```python
   from attune_rag.corpus import CorpusProtocol, RetrievalEntry

   class MyCorpus:
       @property
       def name(self) -> str:
           return "my-corpus"

       @property
       def version(self) -> str:
           return "1.0.0"

       def entries(self):
           yield RetrievalEntry(path="example.md", category="how-to", content="…")

       def get(self, path: str):
           ...
   ```

6. **Run the corpus tests to confirm your changes are correct.**

   ```shell
   pytest -k "corpus"
   ```

## Verify success

The task is complete when:

- `corpus.entries()` yields at least one `RetrievalEntry` with a non-empty `path` and `content`.
- `corpus.get("your/template/path.md")` returns the expected entry rather than `None`.
- `pytest -k "corpus"` passes with no failures or errors.
