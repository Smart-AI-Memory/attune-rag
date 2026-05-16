---
type: concept
name: corpus-concept
feature: corpus
depth: concept
generated_at: 2026-05-16T10:22:21.645528+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus

A corpus is a searchable collection of `RetrievalEntry` objects that the retrieval engine queries to find help templates — think of it as the indexed library that sits between raw markdown files and the engine that serves answers.

## Mental model

Every corpus implementation satisfies `CorpusProtocol`, which requires three things: an `entries()` method that yields all `RetrievalEntry` objects, a `get(path)` method for direct path lookup, and `name` and `version` properties that identify the corpus. Any object that fulfils this contract can be swapped in as a corpus source.

At runtime, the retrieval engine works against `CorpusProtocol` and never talks to a concrete class directly. This means you can back a corpus with bundled templates, a local directory of markdown files, or any other source, without changing the engine.

## Corpus implementations

Two concrete implementations ship out of the box:

**`DirectoryCorpus`** loads every `*.md` file under a root directory (using the glob `**/*.md` by default). It builds two indexes at load time:

- `path_index` — maps each relative file path to its `RetrievalEntry`, used by `get()`.
- `alias_index` — maps each alias declared in a template's frontmatter to an `AliasInfo` record, enabling alternative names to resolve to the same entry.

Its `version` property is a stable SHA-256 fingerprint of the loaded content, so callers can detect when the corpus has changed on disk.

**`AttuneHelpCorpus`** is a thin adapter over the templates bundled with the `attune-help` package. Use `AttuneHelpCorpus.from_attune_help()` to get a pre-configured instance pointing at those bundled templates without specifying a directory path yourself.

## RetrievalEntry

`RetrievalEntry` is a dataclass that represents one template inside a corpus. Its fields are:

| Field | Type | Purpose |
|---|---|---|
| `path` | `str` | Relative path used as the primary lookup key |
| `category` | `str` | Template type (for example, `concept` or `task`) |
| `content` | `str` | Full markdown body served to the user |
| `summary` | `str \| None` | Short description used in compact views |
| `related` | `tuple[str, ...]` | Paths of related entries for cross-linking |
| `aliases` | `tuple[str, ...]` | Alternative names that resolve to this entry |
| `metadata` | `dict[str, Any]` | Arbitrary frontmatter fields |

## Alias resolution and duplicate detection

Aliases let a template be found under more than one name — useful when a concept has a common abbreviation or an older name that users still search for. `DirectoryCorpus` indexes every alias at load time into `alias_index`.

If two templates declare the same alias, `DirectoryCorpus` raises `DuplicateAliasError` immediately, reporting both the conflicting alias and the paths of the two templates. This prevents silent resolution collisions where one template unexpectedly shadows another.

## When each implementation applies

| Situation | Use |
|---|---|
| You want the standard attune-help templates with no configuration | `AttuneHelpCorpus.from_attune_help()` |
| You have a local directory of markdown files | `DirectoryCorpus(root=Path("your/docs"))` |
| You are writing a new corpus source | Implement `CorpusProtocol` and provide `entries()`, `get()`, `name`, and `version` |
