---
type: concept
name: corpus-concept
feature: corpus
depth: concept
generated_at: 2026-06-07T07:12:58.867001+00:00
source_hash: bb71c896e8cc106a568f8d74dd8255cae43c6041f2a7aeb69b2c5dc10e8889f9
status: generated
---

# Corpus

A corpus is a searchable collection of `RetrievalEntry` objects that the retrieval layer queries by path or by iterating all entries — the `CorpusProtocol` defines that contract, and the concrete implementations decide where the content comes from.

## The retrieval contract

`CorpusProtocol` declares the four methods every corpus must expose:

- `entries()` — iterate every `RetrievalEntry` in the collection
- `get(path)` — fetch a single entry by its path string, returning `None` if it doesn't exist
- `name` — a human-readable label for the corpus
- `version` — a stable identifier for the loaded snapshot (for `DirectoryCorpus`, this is a SHA-256 fingerprint of the loaded content)

Any object that satisfies this protocol can be used wherever a corpus is expected, regardless of where its content originates.

## What a RetrievalEntry holds

Each document in a corpus is a `RetrievalEntry` dataclass with the following fields:

| Field | Type | Purpose |
|---|---|---|
| `path` | `str` | Unique identifier and lookup key |
| `category` | `str` | Groups related entries |
| `content` | `str` | Full text of the document |
| `summary` | `str \| None` | Optional short description |
| `related` | `tuple[str, ...]` | Paths of cross-linked entries |
| `aliases` | `tuple[str, ...]` | Alternative names for this entry |
| `metadata` | `dict[str, Any]` | Arbitrary extra data |

The `aliases` field is what makes alternative-name lookup possible. When a corpus indexes its entries, each alias maps back to the entry's `path` through an `AliasInfo` record.

## Concrete corpus implementations

Two implementations ship out of the box.

**`DirectoryCorpus`** loads markdown files from a directory tree. By default it scans `**/*.md` (the `DEFAULT_GLOB` pattern). You can supply:

- `summaries_file` and `cross_links_file` to load pre-computed summaries and relationships from separate files
- `extra_summaries`, `extra_aliases`, and `extra_aliases_file` to inject additional metadata without modifying the source files
- `cache=True` (the default) to avoid rescanning the directory on repeated access

After loading, `DirectoryCorpus` exposes two indexed views in addition to the standard protocol methods:

- `path_index` — a `dict[str, RetrievalEntry]` keyed by relative path
- `alias_index` — a `dict[str, AliasInfo]` keyed by alias string

**`AttuneHelpCorpus`** wraps the bundled attune-help templates. Rather than pointing at a directory you control, you call `AttuneHelpCorpus.from_attune_help()` to get a corpus backed by the templates that ship with the library.

## Alias collision detection

If two entries declare the same alias, `DirectoryCorpus` raises `DuplicateAliasError`, which carries the conflicting `alias`, `first_path`, and `second_path`. Set `warn_alias_overlap=True` (the default) to surface these collisions as warnings during loading rather than letting them go unnoticed.

## How the pieces fit together

```
CorpusProtocol          (interface)
    ├── DirectoryCorpus (reads **/*.md from a root Path)
    │       ├── path_index   → {rel_path: RetrievalEntry}
    │       └── alias_index  → {alias: AliasInfo}
    └── AttuneHelpCorpus (wraps bundled attune-help content)

RetrievalEntry          (the document unit both implementations produce)
DuplicateAliasError     (raised when alias_index would have a collision)
```

Retrieval code works against `CorpusProtocol`, so you can swap `DirectoryCorpus` for `AttuneHelpCorpus` — or a custom implementation — without changing any call sites.
