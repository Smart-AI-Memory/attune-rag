---
type: reference
name: corpus-reference
feature: corpus
depth: reference
generated_at: 2026-05-20T03:23:24.656613+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus reference

Use this API to load, index, and retrieve help templates from a corpus. `CorpusProtocol` defines the interface every corpus must satisfy; `DirectoryCorpus` loads markdown files from disk; `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Constants

| Constant | Type | Value |
|----------|------|-------|
| `DEFAULT_GLOB` | `str` | `'**/*.md'` |

## Classes

| Class | Description |
|-------|-------------|
| `AttuneHelpCorpus` | Loads an attune-help-shaped corpus of templates. |
| `RetrievalEntry` | A single corpus entry. |
| `AliasInfo` | An alias declared by a template, indexed for fast lookup. |
| `DuplicateAliasError` | Raised when two templates declare the same alias. |
| `CorpusProtocol` | Any object that exposes a collection of `RetrievalEntry`. |
| `DirectoryCorpus` | Loads a directory of markdown files as a corpus. |

---

### `CorpusProtocol`

Any object that exposes a collection of `RetrievalEntry`. Implement this protocol to plug a custom corpus into the retrieval pipeline.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry in the corpus. |
| `get` | `self`, `path: str` | `RetrievalEntry \| None` | Returns the entry at the given path, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable name of this corpus. |
| `version` | `str` | Version identifier for this corpus. |

---

### `RetrievalEntry`

`[dataclass]` A single corpus entry, representing one loaded template.

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

### `AttuneHelpCorpus`

Thin adapter over the bundled attune-help templates. Use `from_attune_help()` to construct an instance without managing the underlying adapter directly.

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter` | `HelpCorpusAdapter` | Adapter that provides access to the bundled template data. |

#### Class methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `from_attune_help` | `cls` | `AttuneHelpCorpus` | Constructs an `AttuneHelpCorpus` from the bundled attune-help data. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry in the corpus. |
| `get` | `self`, `path: str` | `RetrievalEntry \| None` | Returns the entry at the given path, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable name of this corpus. |
| `version` | `str` | Version identifier for this corpus. |

---

### `DirectoryCorpus`

Loads a directory of markdown files as a corpus. Supports optional summaries, cross-links, and alias indexing. Entries are cached by default after the first load.

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `Path` | — | Root directory to scan for markdown files. |
| `summaries_file` | `str \| None` | `None` | Path to a file providing per-entry summaries. |
| `cross_links_file` | `str \| None` | `None` | Path to a file providing cross-link relationships. |
| `extra_summaries` | `dict[str, str \| None] \| None` | `None` | Additional summaries supplied directly as a dict. |
| `cache` | `bool` | `True` | Whether to cache loaded entries after the first scan. |
| `glob` | `str` | `DEFAULT_GLOB` | Glob pattern used to discover files under `root`. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry discovered under `root`. |
| `get` | `self`, `path: str` | `RetrievalEntry \| None` | Returns the entry at the given path, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `path_index` | `dict[str, RetrievalEntry]` | `rel_path -> RetrievalEntry` for every loaded template. |
| `alias_index` | `dict[str, AliasInfo]` | `alias -> AliasInfo` for every alias declared in the corpus. |
| `name` | `str` | Human-readable name of this corpus. |
| `version` | `str` | Stable SHA-256 fingerprint of the loaded corpus. |

---

### `AliasInfo`

An alias declared by a template, indexed for fast lookup.

---

### `DuplicateAliasError`

Raised when two templates declare the same alias.

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `alias` | `str` | The alias string that was declared more than once. |
| `first_path` | `str` | Path of the template that first declared the alias. |
| `second_path` | `str` | Path of the template that declared the alias again. |

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

## Tags

`corpus`, `loader`, `markdown`, `attune-help`
