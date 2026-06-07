---
type: reference
name: corpus-reference
feature: corpus
depth: reference
generated_at: 2026-06-07T07:12:58.880357+00:00
source_hash: bb71c896e8cc106a568f8d74dd8255cae43c6041f2a7aeb69b2c5dc10e8889f9
status: generated
---

# Corpus reference

Use the corpus API to load and query collections of retrieval entries. `CorpusProtocol` defines the interface; `DirectoryCorpus` loads markdown files from disk; `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Classes

| Class | Description |
|-------|-------------|
| `CorpusProtocol` | Any object that exposes a collection of `RetrievalEntry`. |
| `DirectoryCorpus` | Loads a directory of markdown files as a corpus. |
| `AttuneHelpCorpus` | Loads an attune-help-shaped corpus of templates. |
| `RetrievalEntry` | A single corpus entry returned by a corpus lookup. |
| `AliasInfo` | An alias declared by a template, indexed for fast lookup. |
| `DuplicateAliasError` | Raised when two templates declare the same alias. |

---

## `CorpusProtocol`

Any object that exposes a collection of `RetrievalEntry`. Implement this protocol to make a custom corpus source compatible with the retrieval engine.

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | — | `Iterable[RetrievalEntry]` | Yields all entries in the corpus. |
| `get` | `path: str` | `RetrievalEntry \| None` | Returns the entry at `path`, or `None` if not found. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable name of the corpus. |
| `version` | `str` | Version identifier for the corpus. |

---

## `DirectoryCorpus`

Loads a directory of markdown files as a corpus. Supports summaries, cross-links, aliases, and optional result caching.

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `Path` | — | Root directory to scan for markdown files. |
| `summaries_file` | `str \| None` | `None` | Path to a file containing per-entry summaries. |
| `cross_links_file` | `str \| None` | `None` | Path to a file containing cross-link rules. |
| `extra_summaries` | `dict[str, str \| None] \| None` | `None` | Additional summaries supplied programmatically. |
| `extra_aliases` | `dict[str, Iterable[str]] \| None` | `None` | Additional aliases supplied programmatically. |
| `extra_aliases_file` | `Path \| str \| None` | `None` | Path to a file containing additional aliases. |
| `cache` | `bool` | `True` | Whether to cache loaded entries. |
| `glob` | `str` | `DEFAULT_GLOB` | Glob pattern used to discover files under `root`. |
| `warn_alias_overlap` | `bool` | `True` | Whether to emit a warning when aliases overlap across entries. |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | — | `Iterable[RetrievalEntry]` | Yields all entries loaded from the directory. |
| `get` | `path: str` | `RetrievalEntry \| None` | Returns the entry at `path`, or `None` if not found. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `path_index` | `dict[str, RetrievalEntry]` | `rel_path -> RetrievalEntry` for every loaded template. |
| `alias_index` | `dict[str, AliasInfo]` | `alias -> AliasInfo` for every alias declared in the corpus. |
| `name` | `str` | Human-readable name of the corpus. |
| `version` | `str` | Stable SHA-256 fingerprint of the loaded corpus. |

---

## `AttuneHelpCorpus`

Thin adapter over the bundled attune-help templates. Use `from_attune_help()` to construct an instance without supplying an adapter directly.

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter` | `HelpCorpusAdapter` | Adapter that provides access to the underlying template store. |

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `from_attune_help` | — | `AttuneHelpCorpus` | Class method. Constructs an `AttuneHelpCorpus` from the bundled attune-help templates. |
| `entries` | — | `Iterable[RetrievalEntry]` | Yields all entries in the corpus. |
| `get` | `path: str` | `RetrievalEntry \| None` | Returns the entry at `path`, or `None` if not found. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable name of the corpus. |
| `version` | `str` | Version identifier for the corpus. |

---

## `RetrievalEntry`

A single corpus entry. Every entry loaded by a corpus implementation is returned as a `RetrievalEntry`.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | — | Relative path that identifies this entry within the corpus. |
| `category` | `str` | — | Category label for the entry. |
| `content` | `str` | — | Full text content of the entry. |
| `summary` | `str \| None` | `None` | Short summary of the entry, if available. |
| `related` | `tuple[str, ...]` | `()` | Paths of related entries. |
| `aliases` | `tuple[str, ...]` | `()` | Alternative lookup keys for this entry. |
| `metadata` | `dict[str, Any]` | `field(default_factory=dict)` | Arbitrary metadata attached to the entry. |

---

## `AliasInfo`

An alias declared by a template, indexed for fast lookup by `DirectoryCorpus.alias_index`.

---

## `DuplicateAliasError`

Raised when two templates declare the same alias.

### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `alias` | `str` | The alias string that appears in both templates. |
| `first_path` | `str` | Path of the first template that declared the alias. |
| `second_path` | `str` | Path of the second template that declared the alias. |

---

## Module constants

| Constant | Type | Value |
|----------|------|-------|
| `DEFAULT_GLOB` | `str` | `'**/*.md'` |

---

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`
