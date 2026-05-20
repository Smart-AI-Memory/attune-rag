---
type: note
name: corpus-note
feature: corpus
depth: note
generated_at: 2026-05-20T03:23:24.677847+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Note: corpus

## Context

The corpus layer provides pluggable loaders for retrieval content. `CorpusProtocol` defines the interface; `DirectoryCorpus` loads markdown files from disk; `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Content

Any object that satisfies `CorpusProtocol` exposes three things: an iterable of `RetrievalEntry` values via `entries()`, a lookup by path via `get()`, and `name` and `version` properties that identify the corpus.

The two built-in implementations differ in where they source content:

- **`DirectoryCorpus`** (`src/attune_rag/corpus/directory.py`) scans a directory tree for markdown files using a configurable glob pattern (default: `**/*.md`). It maintains a `path_index` keyed by relative path and an `alias_index` keyed by alias string. The `version` property is a stable SHA-256 fingerprint of the loaded content. Optional `summaries_file` and `cross_links_file` arguments enrich entries at load time; results are cached by default.

- **`AttuneHelpCorpus`** (`src/attune_rag/corpus/attune_help.py`) is a thin adapter over a `HelpCorpusAdapter` and is the standard way to load the bundled attune-help templates. Use the `from_attune_help()` class method for the default configuration.

Each loaded template is represented as a `RetrievalEntry` dataclass (`src/attune_rag/corpus/base.py`) with the following fields:

| Field | Type | Notes |
|---|---|---|
| `path` | `str` | Corpus-relative path used as the lookup key |
| `category` | `str` | Template category |
| `content` | `str` | Full template text |
| `summary` | `str \| None` | Optional short description |
| `related` | `tuple[str, ...]` | Paths of related entries |
| `aliases` | `tuple[str, ...]` | Alternative lookup keys |
| `metadata` | `dict[str, Any]` | Arbitrary extra data |

Aliases are indexed globally across the corpus. If two templates declare the same alias string, `DirectoryCorpus` raises `DuplicateAliasError`, which carries the alias and both conflicting paths.

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
