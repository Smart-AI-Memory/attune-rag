---
type: error
name: corpus-error
feature: corpus
depth: error
generated_at: 2026-05-20T03:23:24.660675+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus errors

## Common error signatures

Corpus errors typically fall into three categories: duplicate alias declarations, missing or unreadable markdown files on disk, and failed lookups against a corpus that didn't load correctly.

- **`DuplicateAliasError`** — raised during corpus indexing when two templates declare the same alias. The exception carries `alias`, `first_path`, and `second_path`, so you can immediately identify which two files conflict.
- **`KeyError` / `None` return from `get(path)`** — `CorpusProtocol.get()` returns `None` when the requested path doesn't exist in the index. Callers that don't handle `None` will raise a `TypeError` or `AttributeError` downstream.
- **`OSError` / `FileNotFoundError`** — raised by `DirectoryCorpus` when `root` doesn't exist or a matched markdown file can't be read. Also occurs when `summaries_file` or `cross_links_file` paths are invalid.
- **`ValueError`** — can surface during `RetrievalEntry` construction if required fields (`path`, `category`, `content`) are missing or of the wrong type.

## Where errors originate

Each corpus implementation has distinct failure modes. Match the exception to the class before walking the call stack further.

- **`DirectoryCorpus`** (`src/attune_rag/corpus/directory.py`) — errors here are almost always filesystem or indexing problems: `root` doesn't exist, the `glob` pattern (`**/*.md` by default) matches no files, or two templates share an alias. Check `path_index` and `alias_index` after construction to verify what was loaded.
- **`AttuneHelpCorpus`** (`src/attune_rag/corpus/attune_help.py`) — wraps a `HelpCorpusAdapter`; errors here usually mean the bundled templates weren't packaged correctly or `from_attune_help()` couldn't locate the adapter.
- **`RetrievalEntry`** (`src/attune_rag/corpus/base.py`) — a dataclass; construction errors indicate that a markdown file's parsed metadata is missing required fields or contains unexpected types.
- **`DuplicateAliasError`** (`src/attune_rag/corpus/base.py`) — always raised during index-build time, not at query time. The `alias`, `first_path`, and `second_path` attributes identify the conflict precisely.

## How to diagnose

1. **Read `DuplicateAliasError` attributes directly.** If you see this exception, inspect `error.alias`, `error.first_path`, and `error.second_path`. Open both files and remove or rename the duplicate alias declaration in one of them.

2. **Verify the corpus loaded entries.** After constructing a `DirectoryCorpus`, call `len(list(corpus.entries()))` and inspect `corpus.path_index`. An empty index means the `root` path is wrong or the glob matched nothing — confirm `root` exists and contains `*.md` files.

3. **Check `get()` return values before attribute access.** `CorpusProtocol.get(path)` returns `None` for an unknown path. If you're seeing `AttributeError: 'NoneType' object has no attribute '...'`, add a `None` check or log the path value to confirm it matches the keys in `path_index`.

4. **Inspect the `version` fingerprint for caching issues.** `DirectoryCorpus.version` is a SHA-256 fingerprint of the loaded corpus. If you suspect stale data when `cache=True`, compare the `version` value before and after the file change to confirm the corpus was actually reloaded.

5. **Isolate `AttuneHelpCorpus` adapter failures.** If `from_attune_help()` raises, the bundled `HelpCorpusAdapter` may be missing from your installation. Verify the package was installed with its data files intact and that `AttuneHelpCorpus.name` and `AttuneHelpCorpus.version` return non-empty strings before making `entries()` or `get()` calls.

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
