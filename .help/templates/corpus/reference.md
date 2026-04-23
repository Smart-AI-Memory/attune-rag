---
type: reference
feature: corpus
depth: reference
generated_at: 2026-04-23T03:34:16.871793+00:00
source_hash: fbf3871db9ff126e132e66618572aafec8f5d3bb4da48be33dbd1beb2a75d455
status: generated
---

# Corpus reference

Load and access collections of retrievable content from directories or bundled templates.

## Classes

| Class | Description |
|-------|-------------|
| `RetrievalEntry` | A single corpus entry with path, content, and metadata |
| `CorpusProtocol` | Protocol for objects that expose collections of RetrievalEntry |
| `DirectoryCorpus` | Load markdown files from a directory as a corpus |
| `AttuneHelpCorpus` | Access bundled attune-help templates as a corpus |

## RetrievalEntry fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | `str` | | File path or identifier for this entry |
| `category` | `str` | | Classification category for this content |
| `content` | `str` | | Full text content of the entry |
| `summary` | `str \| None` | `None` | Optional summary of the content |
| `related` | `tuple[str, ...]` | `()` | Paths of related entries |
| `metadata` | `dict[str, Any]` | `field(default_factory=dict)` | Additional metadata about the entry |

## CorpusProtocol methods

| Method | Returns | Description |
|--------|---------|-------------|
| `entries()` | `Iterable[RetrievalEntry]` | Return all entries in the corpus |
| `get(path)` | `RetrievalEntry \| None` | Get a specific entry by path |

## CorpusProtocol properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Name identifier for this corpus |
| `version` | `str` | Version string for this corpus |

## DirectoryCorpus methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `root: Path, summaries_file: str \| None = None, cross_links_file: str \| None = None, cache: bool = True, glob: str = DEFAULT_GLOB` | `None` | Initialize corpus from directory |
| `entries` | | `Iterable[RetrievalEntry]` | Return all entries in the corpus |
| `get` | `path: str` | `RetrievalEntry \| None` | Get a specific entry by path |

## DirectoryCorpus properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Name identifier for this corpus |
| `version` | `str` | Version string for this corpus |

## AttuneHelpCorpus methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | | `None` | Initialize corpus with bundled templates |
| `entries` | | `Iterable[RetrievalEntry]` | Return all entries in the corpus |
| `get` | `path: str` | `RetrievalEntry \| None` | Get a specific entry by path |

## AttuneHelpCorpus properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Name identifier for this corpus |
| `version` | `str` | Version string for this corpus |

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DEFAULT_GLOB` | `'**/*.md'` | Default glob pattern for finding markdown files |
