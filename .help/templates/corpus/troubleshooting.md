---
type: troubleshooting
name: corpus-troubleshooting
feature: corpus
depth: troubleshooting
generated_at: 2026-05-20T03:23:24.668708+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Troubleshoot corpus

## Before you start

The corpus module has three main moving parts:

- **`CorpusProtocol`** — the interface that all corpus implementations satisfy. It exposes `entries()`, `get(path)`, `name`, and `version`.
- **`DirectoryCorpus`** — loads `.md` files from disk using a glob pattern (default: `**/*.md`). Builds a `path_index` and an `alias_index` at load time, and optionally caches results.
- **`AttuneHelpCorpus`** — a thin adapter over the bundled attune-help templates. Constructed directly via `AttuneHelpCorpus(adapter)` or through the `from_attune_help()` class method.

Identify which implementation is involved before you start diagnosing.

## Symptom table

| If you observe | Check |
|----------------|-------|
| `DuplicateAliasError` at load time | Two templates declare the same alias. The error exposes `alias`, `first_path`, and `second_path` — open both files and remove or rename the duplicate alias. |
| `get(path)` returns `None` unexpectedly | Confirm the path matches a key in `DirectoryCorpus.path_index`. Keys are relative paths; a leading `/` or wrong separator will cause a miss. |
| `entries()` returns an empty iterable | For `DirectoryCorpus`, verify `root` exists and contains files matching the glob (default `**/*.md`). Print `list(corpus.entries())` to confirm. |
| `AttuneHelpCorpus` raises on construction | Check that the `HelpCorpusAdapter` passed to `__init__` is valid; prefer `AttuneHelpCorpus.from_attune_help()` to let the class build its own adapter. |
| `version` changes between runs unexpectedly | `DirectoryCorpus.version` is a SHA-256 fingerprint of the loaded content. A change means the files on disk changed — check for unintended writes or a stale working directory. |
| Aliases not resolving | Inspect `DirectoryCorpus.alias_index`. Each key is an alias string; its value is an `AliasInfo` object pointing back to the source template. A missing entry means the template's frontmatter alias was not parsed. |
| Slow first load | `DirectoryCorpus` walks the directory and hashes content on first access. Pass `cache=True` (the default) and confirm you are not instantiating a new `DirectoryCorpus` on every request. |

## Step-by-step diagnosis

1. **Reproduce the failure in isolation.**
   Reduce the call to its minimum required arguments. For `DirectoryCorpus`, that is just `root`:

   ```python
   from pathlib import Path
   from attune_rag.corpus import DirectoryCorpus

   corpus = DirectoryCorpus(root=Path("path/to/templates"))
   print(list(corpus.entries()))
   print(corpus.version)
   ```

   For `AttuneHelpCorpus`:

   ```python
   from attune_rag.corpus import AttuneHelpCorpus

   corpus = AttuneHelpCorpus.from_attune_help()
   print(corpus.name, corpus.version)
   ```

   Confirm the failure occurs before adding complexity back.

2. **Inspect the indexes.**
   If `get()` or alias resolution is misbehaving, print the internal indexes before assuming the content is wrong:

   ```python
   # DirectoryCorpus only
   print(corpus.path_index.keys())   # all loaded relative paths
   print(corpus.alias_index.keys())  # all declared aliases
   ```

   A missing key here means the file was not loaded or its frontmatter was not parsed correctly.

3. **Check for duplicate aliases.**
   If you see `DuplicateAliasError`, the exception message includes the alias string, `first_path`, and `second_path`. Open both files and deduplicate:

   ```
   DuplicateAliasError: alias='foo', first_path='a/one.md', second_path='b/two.md'
   ```

   Remove or rename the alias in one of the two templates.

4. **Verify the glob pattern.**
   `DirectoryCorpus` defaults to `**/*.md`. If your templates use a different extension or live in an unexpected subdirectory, override the glob:

   ```python
   corpus = DirectoryCorpus(root=Path("templates"), glob="**/*.markdown")
   ```

   Run `list(Path("templates").glob("**/*.md"))` directly to confirm which files Python finds.

5. **Run the corpus tests.**
   Before modifying any code, run the existing test suite to establish a baseline:

   ```bash
   pytest -k "corpus" -v
   ```

   A failing test that exercises your exact path gives you a reproducible fixture to work against.

## Common fixes

- **Duplicate alias.** Edit the offending template identified by `DuplicateAliasError.first_path` or `second_path` and remove the conflicting alias from its frontmatter.

- **Path mismatch in `get()`.** Normalize the path you pass to `get()` to match the relative-path keys in `path_index`:

  ```python
  entry = corpus.get("how-to/deploy.md")   # correct: relative, no leading slash
  entry = corpus.get("/how-to/deploy.md")  # wrong: leading slash causes None
  ```

- **Empty corpus from wrong root.** If `entries()` is empty, verify the `root` argument is the directory that *contains* the markdown files, not a parent directory:

  ```python
  corpus = DirectoryCorpus(root=Path("src/attune_rag/corpus/templates"))
  ```

- **Stale cached state.** If the corpus was cached at module import time and templates have changed on disk, reinstantiate `DirectoryCorpus` or restart the process. There is no explicit cache-invalidation API; the `version` SHA-256 fingerprint tells you whether disk content has drifted.

- **Wrong adapter for `AttuneHelpCorpus`.** If constructing `AttuneHelpCorpus` directly raises, switch to the factory method, which handles adapter construction internally:

  ```python
  corpus = AttuneHelpCorpus.from_attune_help()
  ```

- **Dependency or environment drift.** If the corpus loaded correctly previously but no longer does, run:

  ```bash
  pip show attune-rag
  ```

  and confirm the installed version matches your expectations. A version upgrade may have changed the bundled template paths that `AttuneHelpCorpus` relies on.

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
