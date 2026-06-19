---
type: task
name: corpus-task
feature: corpus
depth: task
generated_at: 2026-06-07T07:12:58.874153+00:00
source_hash: bb71c896e8cc106a568f8d74dd8255cae43c6041f2a7aeb69b2c5dc10e8889f9
status: generated
---

# Work with a corpus

Use a corpus implementation when you need to load and query a collection of retrieval entries — either from a directory of Markdown files (`DirectoryCorpus`), from the bundled attune-help templates (`AttuneHelpCorpus`), or from a custom source that satisfies `CorpusProtocol`.

## Prerequisites

- Read access to the project source under `src/attune_rag/corpus/`
- A working Python environment with the package installed

## Choose a corpus implementation

The three concrete options map to distinct use cases:

| Class | Use when |
|---|---|
| `DirectoryCorpus` | You have a local directory of Markdown files to load as a corpus |
| `AttuneHelpCorpus` | You want to query the bundled attune-help templates directly |
| Custom `CorpusProtocol` | You need a corpus backed by a source other than the filesystem |

## Load a `DirectoryCorpus`

1. **Import the class** from the public API:

   ```python
   from corpus import DirectoryCorpus
   ```

2. **Instantiate it** with the path to your Markdown directory:

   ```python
   corpus = DirectoryCorpus(root=Path("docs/"))
   ```

   Pass optional arguments to refine loading behavior:
   - `summaries_file` — path to a file that provides per-entry summaries
   - `cross_links_file` — path to a file that provides cross-link relationships
   - `extra_aliases` or `extra_aliases_file` — additional aliases beyond what templates declare
   - `glob` — override the default `**/*.md` pattern to match a different set of files
   - `cache=False` — disable caching if you need entries to reflect live file changes
   - `warn_alias_overlap=False` — suppress warnings when aliases collide across templates

3. **Iterate over entries** to access all loaded templates:

   ```python
   for entry in corpus.entries():
       print(entry.path, entry.category)
   ```

4. **Retrieve a single entry by path** using `get`:

   ```python
   entry = corpus.get("tasks/my-template.md")
   ```

   `get` returns `None` if no entry matches the path.

5. **Look up entries and aliases by index** for fast access:

   ```python
   entry = corpus.path_index["tasks/my-template.md"]
   alias_info = corpus.alias_index["my-alias"]
   ```

## Load an `AttuneHelpCorpus`

1. **Import and instantiate** using the class method:

   ```python
   from corpus import AttuneHelpCorpus

   corpus = AttuneHelpCorpus.from_attune_help()
   ```

2. **Query entries** the same way as `DirectoryCorpus` — both implement `CorpusProtocol`:

   ```python
   for entry in corpus.entries():
       print(entry.path)

   entry = corpus.get("concepts/some-concept.md")
   ```

## Implement a custom `CorpusProtocol`

1. **Import the protocol**:

   ```python
   from corpus import CorpusProtocol, RetrievalEntry
   ```

2. **Define a class** that implements the four required members:

   ```python
   class MyCorpus:
       def entries(self) -> Iterable[RetrievalEntry]:
           ...

       def get(self, path: str) -> RetrievalEntry | None:
           ...

       @property
       def name(self) -> str:
           ...

       @property
       def version(self) -> str:
           ...
   ```

   Your class does not need to subclass `CorpusProtocol` — any object that satisfies the interface is accepted wherever a `CorpusProtocol` is expected.

3. **Construct `RetrievalEntry` objects** for each item your corpus exposes. The required fields are `path`, `category`, and `content`:

   ```python
   entry = RetrievalEntry(
       path="custom/my-entry.md",
       category="tasks",
       content="# My entry\n\nContent here.",
       summary="A short description",
       aliases=("my-entry", "entry-alias"),
   )
   ```

## Handle `DuplicateAliasError`

When two templates in the same corpus declare the same alias, `DirectoryCorpus` raises `DuplicateAliasError` (unless `warn_alias_overlap=False` is set, in which case it warns instead). Catch it if you load corpora programmatically:

```python
from corpus import DirectoryCorpus, DuplicateAliasError

try:
    corpus = DirectoryCorpus(root=Path("docs/"))
except DuplicateAliasError as e:
    print(f"Alias '{e.alias}' is claimed by both {e.first_path} and {e.second_path}")
```

## Verify the result

Run the corpus test suite to confirm everything loads correctly:

```
pytest -k "corpus"
```

A passing run confirms that all entries load without alias conflicts, `get` resolves paths correctly, and `entries()` returns the expected `RetrievalEntry` objects. You can also do a quick sanity check in a Python session:

```python
corpus = DirectoryCorpus(root=Path("docs/"))
assert corpus.name  # non-empty string
assert any(True for _ in corpus.entries())  # at least one entry loaded
print(f"Loaded corpus '{corpus.name}' at version {corpus.version}")
```

## Unresolved references

> Auto-generated by attune-author fact-check. Review and either
> fix the source code, fix this doc, or add an override.

| Location | Severity | Issue |
|---|---|---|
| Line 138 (code fence) | error | `from corpus import …` — module not importable |
