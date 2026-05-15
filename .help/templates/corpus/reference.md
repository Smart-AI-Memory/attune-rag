---
type: reference
name: corpus-reference
feature: corpus
depth: reference
generated_at: 2026-05-15T20:02:03.235684+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus reference

Load and query collections of help templates. `CorpusProtocol` defines the interface; `DirectoryCorpus` loads markdown files from disk; `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Classes

| Class | Description |
|-------|-------------|
| `CorpusProtocol` | Protocol any corpus implementation must satisfy — defines `entries()`, `get()`, `name`, and `version`. |
| `RetrievalEntry` | Single corpus entry holding a template's path, content, summary, related paths, aliases, and metadata. |
| `AliasInfo` | Alias declared by a template, indexed for fast lookup. |
| `DuplicateAliasError` | Raised when two templates declare the same alias. |
| `DirectoryCorpus` | Loads a directory of markdown files as a corpus, with optional summaries and cross-links. |
| `AttuneHelpCorpus` | Thin adapter over the bundled attune-help templates. |

---

### `CorpusProtocol`

Any object that exposes a collection of `RetrievalEntry`. Implement this protocol to plug in a custom corpus.

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry in the corpus. |
| `get` | `self, path: str` | `RetrievalEntry \| None` | Returns the entry at `path`, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable name of the corpus. |
| `version` | `str` | Version identifier for the corpus. |

---

### `RetrievalEntry`

`[dataclass]` A single corpus entry. Carries all data the retrieval layer needs to serve or rank a template.

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

An alias declared by a template, indexed for fast lookup.

---

### `DuplicateAliasError`

Raised when two templates declare the same alias.

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `alias` | `str` | The alias string that appears in both templates. |
| `first_path` | `str` | Path of the first template that declared the alias. |
| `second_path` | `str` | Path of the second template that declared the alias. |

---

### `DirectoryCorpus`

Loads a directory of markdown files as a corpus. Supports optional summaries, cross-links, and caching.

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `root` | `Path` | — | Root directory to scan for templates. |
| `summaries_file` | `str \| None` | `None` | Path to a JSON file of pre-computed summaries. |
| `cross_links_file` | `str \| None` | `None` | Path to a JSON file of cross-link data. |
| `extra_summaries` | `dict[str, str \| None] \| None` | `None` | Additional summaries to merge over the summaries file. |
| `cache` | `bool` | `True` | Whether to cache loaded entries in memory. |
| `glob` | `str` | `DEFAULT_GLOB` | Glob pattern used to discover files under `root`. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every loaded `RetrievalEntry`. |
| `get` | `self, path: str` | `RetrievalEntry \| None` | Returns the entry at `path`, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `path_index` | `dict[str, RetrievalEntry]` | `rel_path -> RetrievalEntry` for every loaded template. |
| `alias_index` | `dict[str, AliasInfo]` | `alias -> AliasInfo` for every alias declared in the corpus. |
| `name` | `str` | Human-readable name of this corpus. |
| `version` | `str` | Stable SHA-256 fingerprint of the loaded corpus. |

---

### `AttuneHelpCorpus`

Thin adapter over the bundled attune-help templates. Use `from_attune_help()` to construct an instance from the default bundle.

#### Constructor

| Parameter | Type | Description |
|-----------|------|-------------|
| `adapter` | `HelpCorpusAdapter` | Adapter that provides access to the bundled templates. |

#### Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `from_attune_help` | `cls` | `AttuneHelpCorpus` | Class method. Constructs an `AttuneHelpCorpus` from the bundled attune-help templates. |
| `entries` | `self` | `Iterable[RetrievalEntry]` | Yields every entry in the attune-help corpus. |
| `get` | `self, path: str` | `RetrievalEntry \| None` | Returns the entry at `path`, or `None` if not found. |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable name of this corpus. |
| `version` | `str` | Version identifier for the bundled corpus. |

---

## Constants

| Constant | Type | Value | Description |
|----------|------|-------|-------------|
| `DEFAULT_GLOB` | `str` | `**/*.md` | Default glob pattern used by `DirectoryCorpus` to discover markdown files. |

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

## Tags

`corpus`, `loader`, `markdown`, `attune-help`
