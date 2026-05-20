---
type: faq
name: corpus-faq
feature: corpus
depth: faq
generated_at: 2026-05-20T03:23:24.671095+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Corpus FAQ

## What is a corpus?

A corpus is a collection of `RetrievalEntry` objects that your code can query by path. The `CorpusProtocol` defines the interface (`entries()`, `get()`, `name`, `version`). Two implementations ship out of the box: `DirectoryCorpus` loads markdown files from a directory on disk, and `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Which corpus class should I use?

It depends on where your content lives:

- Use `DirectoryCorpus` when your markdown files are in a directory you control. Pass the root `Path` and optionally a `summaries_file`, `cross_links_file`, or a custom `glob` pattern (default: `**/*.md`).
- Use `AttuneHelpCorpus` when you want to query the bundled attune-help templates. Call the class method `AttuneHelpCorpus.from_attune_help()` rather than constructing it directly.
- Implement `CorpusProtocol` directly if neither built-in class fits your source.

## How do I load the bundled attune-help templates?

Call `AttuneHelpCorpus.from_attune_help()`. This is the standard way to get an `AttuneHelpCorpus` instance without wiring up a `HelpCorpusAdapter` yourself.

## What does a single corpus entry look like?

Each entry is a `RetrievalEntry` dataclass with these fields:

| Field | Type | Notes |
|---|---|---|
| `path` | `str` | Unique identifier for the entry |
| `category` | `str` | Required |
| `content` | `str` | Required |
| `summary` | `str \| None` | Optional short description |
| `related` | `tuple[str, ...]` | Paths of related entries |
| `aliases` | `tuple[str, ...]` | Alternative lookup keys |
| `metadata` | `dict[str, Any]` | Arbitrary extra data |

## How do I look up an entry by path?

Call `corpus.get(path)`, which returns a `RetrievalEntry` or `None` if the path is not found. To iterate over all entries, use `corpus.entries()`.

## What happens if two templates declare the same alias?

`DirectoryCorpus` raises a `DuplicateAliasError` at load time. The error message includes the alias and both conflicting paths (`first_path`, `second_path`), so you can find and resolve the conflict quickly.

## How do I find all aliases in a `DirectoryCorpus`?

Read the `alias_index` property, which returns a `dict[str, AliasInfo]` mapping every declared alias to its `AliasInfo`. For a path-keyed view of all loaded entries, use `path_index` instead.

## How does `DirectoryCorpus` identify its version?

The `version` property returns a stable SHA-256 fingerprint of the loaded corpus content, so you can detect when the corpus has changed without comparing entries individually.

## How do I debug a corpus that isn't loading entries correctly?

Run the relevant tests first: `pytest -k "corpus" -v`. If they pass but your code still fails, check these things in order:

1. Confirm your `root` path exists and contains `.md` files matching your `glob` pattern (default: `**/*.md`).
2. Inspect `corpus.path_index` to see which entries were actually loaded.
3. If aliases are involved, check `corpus.alias_index` for unexpected mappings or watch for `DuplicateAliasError` at construction time.
4. Add a `logger.debug` statement at the suspected failure point and re-run with logging enabled.

## Where are the source files?

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
