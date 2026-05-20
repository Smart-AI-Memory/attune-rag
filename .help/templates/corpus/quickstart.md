---
type: quickstart
name: corpus-quickstart
feature: corpus
depth: quickstart
generated_at: 2026-05-20T03:23:24.673388+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Quickstart: corpus

The corpus module gives you a consistent interface for loading and retrieving markdown-based help templates. `DirectoryCorpus` loads any directory of markdown files; `AttuneHelpCorpus` wraps the bundled attune-help templates; `CorpusProtocol` defines the interface both implement.

```python
from attune_rag.corpus import AttuneHelpCorpus

corpus = AttuneHelpCorpus.from_attune_help()
for entry in corpus.entries():
    print(entry.path, entry.category)
```

Expected output (paths will vary by installation):

```
quickstart/corpus-quickstart  quickstart
reference/corpus-reference    reference
...
```

## Prerequisites

- The package is installed in your Python environment.
- For `DirectoryCorpus`: a directory of `.md` files you want to load.

## Steps

1. **Load a corpus.** Use `AttuneHelpCorpus.from_attune_help()` for the bundled templates, or point `DirectoryCorpus` at your own markdown directory:

   ```python
   from pathlib import Path
   from attune_rag.corpus import DirectoryCorpus

   corpus = DirectoryCorpus(root=Path("docs/help"))
   ```

2. **Retrieve a specific entry by path.** Call `corpus.get(path)` with the relative path of the template you want. It returns a `RetrievalEntry` or `None` if the path is not found:

   ```python
   entry = corpus.get("quickstart/corpus-quickstart")
   if entry:
       print(entry.summary)
       print(entry.content[:200])
   ```

3. **Inspect corpus metadata.** Check `corpus.name` and `corpus.version` to confirm what was loaded. For `DirectoryCorpus`, `version` is a stable SHA-256 fingerprint of the loaded files:

   ```python
   print(corpus.name)     # e.g. "attune-help"
   print(corpus.version)  # e.g. "a3f9c2..."
   ```

4. **Watch for duplicate aliases.** If two templates declare the same alias, `DirectoryCorpus` raises `DuplicateAliasError` on load. Catch it to surface the conflict:

   ```python
   from attune_rag.corpus import DirectoryCorpus, DuplicateAliasError

   try:
       corpus = DirectoryCorpus(root=Path("docs/help"))
   except DuplicateAliasError as exc:
       print(f"Alias '{exc.alias}' claimed by {exc.first_path} and {exc.second_path}")
   ```

## Expected result

After a successful load, `corpus.entries()` yields `RetrievalEntry` objects. Each entry exposes `path`, `category`, `content`, and optional fields including `summary`, `related`, `aliases`, and `metadata`.

## Next

Read the `CorpusProtocol` reference to understand how to write your own corpus implementation that works anywhere a `DirectoryCorpus` or `AttuneHelpCorpus` is accepted.

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
