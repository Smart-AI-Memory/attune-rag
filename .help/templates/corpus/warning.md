---
type: warning
name: corpus-warning
feature: corpus
depth: warning
generated_at: 2026-05-20T03:23:24.666475+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus cautions

## What to watch for

The corpus module provides three entry points: `CorpusProtocol` defines the interface, `DirectoryCorpus` loads markdown files from disk, and `AttuneHelpCorpus` wraps the bundled attune-help templates. The risks below apply when you build, extend, or swap out a corpus implementation.

## Risk areas

### Duplicate aliases cause a hard failure at load time

If two templates in a corpus declare the same alias string, `DirectoryCorpus` raises `DuplicateAliasError(alias, first_path, second_path)` before any entries are returned. This error surfaces the two conflicting paths, so you can resolve the collision directly. To avoid it, audit the `aliases` field of every `RetrievalEntry` you add to a shared corpus — aliases must be globally unique within a corpus instance.

### `DirectoryCorpus` caches eagerly by default

`DirectoryCorpus` accepts a `cache` parameter (default `True`). When caching is enabled, the `path_index` and `alias_index` properties are built once from the files on disk. If you modify, add, or remove markdown files after the corpus is constructed, those changes are invisible until you create a new `DirectoryCorpus` instance. Pass `cache=False` only when you need live reloading during development — it adds I/O overhead on every call to `entries()`.

### The `DEFAULT_GLOB` pattern silently skips non-markdown files

`DirectoryCorpus` defaults to `**/*.md`. Any file in the root directory that doesn't match this pattern is ignored without warning. If entries you expect are missing from a corpus, verify that the files have `.md` extensions and that you haven't overridden `glob` with a pattern that excludes them.

### `RetrievalEntry` metadata is an open `dict` with no schema enforcement

The `metadata` field of `RetrievalEntry` accepts any `dict[str, Any]`. Code that reads from `metadata` must handle missing keys defensively — the corpus loader does not validate the contents. Undocumented keys added by one part of the system can be silently dropped or misread by another.

### `AttuneHelpCorpus` depends on the bundled adapter, not on disk

`AttuneHelpCorpus.from_attune_help()` constructs the corpus from a `HelpCorpusAdapter` that wraps the bundled templates. It does not read from the filesystem at call time. If you expect to use a customized template directory, `AttuneHelpCorpus` is the wrong choice — use `DirectoryCorpus` with your own `root` path instead.

### Custom `CorpusProtocol` implementations must satisfy both `entries()` and `get()`

`CorpusProtocol` requires `entries()`, `get(path)`, `name`, and `version`. A common mistake is implementing `entries()` correctly but returning `None` from `get()` for paths that `entries()` yields. Callers that use `get()` for fast lookup will silently receive `None` rather than raising an error, which can produce incorrect retrieval results downstream.

## How to avoid problems

1. **Check `DuplicateAliasError` during corpus construction.** Wrap `DirectoryCorpus` instantiation in a try/except when loading user-supplied template directories. The error message includes both conflicting paths, making the fix straightforward.

2. **Reconstruct `DirectoryCorpus` when files change.** Don't hold a long-lived instance across file modifications in tests or tooling. Create a new instance instead of relying on the cached index to refresh.

3. **Validate `metadata` contents at the point of insertion.** If your pipeline writes custom keys into `RetrievalEntry.metadata`, define and document the expected schema in one place rather than scattering key-name assumptions across consumers.

4. **Depend only on `CorpusProtocol`'s public interface.** Private helpers (names starting with `_`) can change without notice. Code that calls `_`-prefixed attributes on any corpus class is likely to break during refactors.

5. **Run `pytest -k "corpus"` before integrating changes.** The corpus index is shared across retrieval; a regression in alias lookup or path resolution can silently degrade answer quality rather than raising an exception.

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
