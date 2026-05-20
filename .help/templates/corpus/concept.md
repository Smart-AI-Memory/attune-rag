---
type: concept
name: corpus-concept
feature: corpus
depth: concept
generated_at: 2026-05-20T03:23:24.644946+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus

A corpus is a searchable collection of `RetrievalEntry` objects that your retrieval pipeline queries to find relevant help templates.

## Mental model

The corpus layer separates *what you load* from *how you load it*. `CorpusProtocol` defines the contract every corpus must satisfy — `entries()`, `get(path)`, `name`, and `version` — so the rest of the codebase never depends on a specific source format.

Two concrete implementations ship out of the box:

- **`DirectoryCorpus`** scans a directory for markdown files matching `**/*.md` (configurable via `glob`). It builds a `path_index` (relative path → entry) and an `alias_index` (alias string → `AliasInfo`) for fast lookup, and computes a stable SHA-256 `version` fingerprint over the loaded files. Optional `summaries_file` and `cross_links_file` arguments let you augment entries without editing the markdown source.
- **`AttuneHelpCorpus`** is a thin adapter over the bundled attune-help templates. Rather than reading the filesystem at runtime, you construct it via `AttuneHelpCorpus.from_attune_help()`, which wires up the bundled `HelpCorpusAdapter` for you.

## RetrievalEntry — the unit of content

Every piece of content in a corpus is a `RetrievalEntry` dataclass with these fields:

| Field | Type | Purpose |
|-------|------|---------|
| `path` | `str` | Unique identifier for the entry (typically its relative file path) |
| `category` | `str` | Groups related entries for filtering |
| `content` | `str` | Full markdown body of the template |
| `summary` | `str \| None` | Optional short description used in search result previews |
| `related` | `tuple[str, ...]` | Paths of entries the retriever should consider alongside this one |
| `aliases` | `tuple[str, ...]` | Alternative names that resolve to this entry |
| `metadata` | `dict[str, Any]` | Arbitrary key/value data for downstream consumers |

## Alias indexing and conflict detection

When `DirectoryCorpus` loads a template that declares `aliases`, it indexes each alias in `alias_index` for O(1) lookup. If two templates declare the same alias string, `DuplicateAliasError` is raised immediately, identifying both the conflicting alias and the paths of the two templates involved. This prevents silent resolution ambiguity at load time rather than at query time.

## Integration points

Any object that satisfies `CorpusProtocol` can be dropped in wherever a corpus is expected. To implement your own source, expose the four protocol members:

| Member | Description |
|--------|-------------|
| `entries() -> Iterable[RetrievalEntry]` | Yields every entry in the corpus |
| `get(path: str) -> RetrievalEntry \| None` | Returns a single entry by path, or `None` |
| `name` | Human-readable label for the corpus |
| `version` | Opaque string that changes when content changes (used for cache invalidation) |
