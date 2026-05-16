---
type: task
name: corpus-task
feature: corpus
depth: task
generated_at: 2026-05-16T10:22:21.651757+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Work with corpus

Use the corpus module when you need to load, extend, or replace a collection of retrieval entries — `CorpusProtocol` defines the interface, `DirectoryCorpus` loads markdown files from disk, and `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Prerequisites

- Read access to the project source code
- Basic familiarity with Python dataclasses and protocols

## Steps

1. **Identify the right entry point.**
   Choose the corpus class that matches your use case:
   - `AttuneHelpCorpus` (`src/attune_rag/corpus/attune_help.py`) — use this when you want the bundled attune-help templates. Call `AttuneHelpCorpus.from_attune_help()` for the default instance.
   - `DirectoryCorpus` (`src/attune_rag/corpus/directory.py`) — use this when you want to load your own markdown files from a directory. Pass a `Path` to the constructor; optionally supply `summaries_file` or `cross_links_file`.
   - `CorpusProtocol` (`src/attune_rag/corpus/base.py`) — implement this protocol directly when neither existing class fits.

2. **Retrieve entries.**
   Call `entries()` on your corpus instance to iterate over all `RetrievalEntry` objects, or call `get(path)` to fetch a single entry by its relative path. Each `RetrievalEntry` exposes `path`, `category`, `content`, and optional fields: `summary`, `related`, `aliases`, and `metadata`.

3. **Extend rather than modify base classes.**
   If you need custom loading behaviour, subclass `DirectoryCorpus` or `AttuneHelpCorpus` and override `entries()` or `get()`. Avoid editing `CorpusProtocol` or `RetrievalEntry` directly — changes there affect every corpus implementation.

4. **Handle alias conflicts.**
   If two templates declare the same alias, `DirectoryCorpus` raises `DuplicateAliasError` with the conflicting `alias`, `first_path`, and `second_path`. Inspect the `alias_index` property on `DirectoryCorpus` to audit all registered aliases before loading a new corpus root.

5. **Run the tests.**
   Verify your changes with:
   ```
   pytest -k "corpus"
   ```

## Key files

| File | Purpose |
|---|---|
| `src/attune_rag/corpus/__init__.py` | Public exports (`__all__`) |
| `src/attune_rag/corpus/base.py` | `CorpusProtocol`, `RetrievalEntry`, `AliasInfo`, `DuplicateAliasError` |
| `src/attune_rag/corpus/directory.py` | `DirectoryCorpus` — markdown-from-disk loader |
| `src/attune_rag/corpus/attune_help.py` | `AttuneHelpCorpus` — bundled attune-help adapter |

## Verify the task worked

After running `pytest -k "corpus"`, all tests pass with no errors. If you added a new corpus class, confirm that `entries()` returns at least one `RetrievalEntry` and that `get(path)` returns the expected entry for a known path. For `DirectoryCorpus`, check the `version` property — it returns a stable SHA-256 fingerprint of the loaded corpus, so re-loading the same directory produces the same value.
