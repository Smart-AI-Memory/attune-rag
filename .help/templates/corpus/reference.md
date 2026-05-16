---
type: reference
name: corpus-reference
feature: corpus
depth: reference
generated_at: 2026-05-16T10:22:21.656713+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus reference

Use the corpus API to load collections of templates for retrieval. `CorpusProtocol` defines the interface; `DirectoryCorpus` loads markdown files from disk; `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Classes

| Class | Description |
|-------|-------------|
| `CorpusProtocol` | Protocol that any corpus implementation must satisfy — defines `entries()`, `get()`, `name`, and `version`. |
| `AttuneHelpCorpus` | Loads an attune-help-shaped corpus of templates. |
| `DirectoryCorpus` | Loads a directory of markdown files as a corpus. |
| `RetrievalEntry` | A single corpus entry. |
| `AliasInfo` | An alias declared by a template, indexed for fast lookup. |
| `DuplicateAliasError` | Raised when two templates declare the same alias. |

---

### `CorpusProtocol`

Any object that exposes a collection of `RetrievalEntry`. Implement this protocol to create a custom corpus backend.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry in the corpus. |
| `get` | `self, path: str` | `RetrievalEntry \| None` | Returns the entry for the given path, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Name of this corpus. |
| `version` | `str` | Version identifier for this corpus. |

---

### `AttuneHelpCorpus`

Thin adapter over the bundled attune-help templates. Use `from_attune_help()` to construct an instance from the built-in corpus without supplying an adapter manually.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `self, adapter: HelpCorpusAdapter` | `None` | Initializes the corpus with the given adapter. |
| `from_attune_help` | `cls` | `AttuneHelpCorpus` | Constructs an instance from the bundled attune-help corpus. |
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry in the corpus. |
| `get` | `self, path: str` | `RetrievalEntry \| None` | Returns the entry for the given path, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Name of this corpus. |
| `version` | `str` | Version identifier for this corpus. |

---

### `DirectoryCorpus`

Loads a directory of markdown files as a corpus. Supports optional summary and cross-link overlays, alias indexing, and result caching.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `self, root: Path, summaries_file: str \| None = None, cross_links_file: str \| None = None, extra_summaries: dict[str, str \| None] \| None = None, cache: bool = True, glob: str = DEFAULT_GLOB` | `None` | Loads markdown files under `root` matching `glob`. |
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry in the corpus. |
| `get` | `self, path: str` | `RetrievalEntry \| None` | Returns the entry for the given path, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `path_index` | `dict[str, RetrievalEntry]` | `rel_path -> RetrievalEntry` for every loaded template. |
| `alias_index` | `dict[str, AliasInfo]` | `alias -> AliasInfo` for every alias declared in the corpus. |
| `name` | `str` | Name of this corpus. |
| `version` | `str` | Stable SHA-256 fingerprint of the loaded corpus. |

---

### `RetrievalEntry` [dataclass]

A single corpus entry returned by `entries()` or `get()`.

#### Fields

| Field | Type | Default |
|-------|------|---------|
| `path` | `str` | — |
| `category` | `str` | — |
| `content` | `str` | — |
| `summary` | `str \| None` | `None` |
| `related` | `tuple[str, ...]` | `()` |
| `aliases` | `tuple[str, ...]` | `()` |
| `metadata` | `dict[str, Any]` | `field(default_factory=dict)` |

---

### `AliasInfo`

An alias declared by a template, indexed for fast lookup. Used internally by `DirectoryCorpus.alias_index`.

---

### `DuplicateAliasError`

Raised when two templates declare the same alias.

#### Constructor

| Method | Parameters | Returns |
|--------|------------|---------|
| `__init__` | `self, alias: str, first_path: str, second_path: str` | `None` |

---

## Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `DEFAULT_GLOB` | `str` | `'**/*.md'` | Default glob pattern used by `DirectoryCorpus` to discover markdown files. |

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

## Tags

`corpus`, `loader`, `markdown`, `attune-help`
