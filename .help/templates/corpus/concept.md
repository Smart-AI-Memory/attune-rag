---
type: concept
name: corpus-concept
feature: corpus
depth: concept
generated_at: 2026-05-15T20:02:03.228922+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus

A corpus is a searchable collection of help templates that the retrieval engine queries at runtime — think of it as the indexed library from which attune-help serves answers.

## Mental model

The corpus layer has three moving parts that fit together in a pipeline:

1. **A loader** reads templates from some source — a directory of markdown files, or the bundled attune-help package — and produces a flat stream of `RetrievalEntry` objects.
2. **The protocol** (`CorpusProtocol`) defines the contract every loader must satisfy: expose `entries()`, support point lookup via `get(path)`, and report a `name` and `version`.
3. **The retrieval engine** calls those methods without caring which loader is underneath — swapping `DirectoryCorpus` for `AttuneHelpCorpus` requires no changes elsewhere.

This separation means you can point the engine at your own markdown directory during development and at the production attune-help bundle in CI, using the same interface both times.

## Core components

### `CorpusProtocol`

The structural interface that any corpus implementation must satisfy. It requires four members:

| Member | Description |
|--------|-------------|
| `entries()` | Returns every `RetrievalEntry` in the corpus as an iterable. |
| `get(path)` | Returns a single `RetrievalEntry` by its relative path, or `None`. |
| `name` | A human-readable identifier for the corpus. |
| `version` | A stable SHA-256 fingerprint of the loaded content (on `DirectoryCorpus`), used to detect drift. |

### `RetrievalEntry`

The atomic unit of corpus content. Each entry carries the fields the retrieval engine needs to match a query to a template:

| Field | Type | Purpose |
|-------|------|---------|
| `path` | `str` | Relative path used as the primary key for `get()`. |
| `category` | `str` | Groups entries by template type (e.g., `concepts`, `tasks`). |
| `content` | `str` | The full markdown body. |
| `summary` | `str \| None` | Short description used in compact views. |
| `related` | `tuple[str, ...]` | Paths of cross-linked entries. |
| `aliases` | `tuple[str, ...]` | Alternative lookup keys (see alias indexing below). |
| `metadata` | `dict` | Arbitrary frontmatter fields passed through unchanged. |

### `DirectoryCorpus`

Loads any directory of markdown files matching `**/*.md` (the default glob, which you can override). After loading, it exposes two indexes for fast lookup:

- **`path_index`** — maps each file's relative path to its `RetrievalEntry`.
- **`alias_index`** — maps each declared alias to an `AliasInfo` object. If two templates claim the same alias, `DirectoryCorpus` raises `DuplicateAliasError` immediately, so conflicts surface at load time rather than at query time.

The `version` property returns a stable SHA-256 fingerprint of the loaded content, so callers can detect when the corpus has changed without re-reading every file.

### `AttuneHelpCorpus`

A thin adapter that loads the bundled attune-help templates without requiring you to know their location on disk. Use the `from_attune_help()` class method to construct one:

```python
corpus = AttuneHelpCorpus.from_attune_help()
entry = corpus.get("concepts/task-template-design-patterns.md")
```

`AttuneHelpCorpus` satisfies `CorpusProtocol`, so you can pass it anywhere a corpus is expected.

## Alias indexing and conflict detection

Templates can declare aliases — alternative names under which the retrieval engine can find them. `DirectoryCorpus` indexes every alias at load time. If two templates declare the same alias, the loader raises `DuplicateAliasError` with the alias string and both conflicting paths, letting you fix the collision before it causes silent misdirection at query time.

## When the corpus layer matters

You interact with the corpus layer when you:

- **Add a new template source** — implement `CorpusProtocol` and pass your corpus to the retrieval engine.
- **Debug a missing entry** — check whether `get(path)` returns `None`, which means the path doesn't match any loaded file.
- **Detect content drift** — compare `DirectoryCorpus.version` across runs to know whether the on-disk templates have changed.
- **Resolve a `DuplicateAliasError`** — find the two templates listed in the error and remove or rename one alias.
