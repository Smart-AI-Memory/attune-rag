---
type: task
name: corpus-task
feature: corpus
depth: task
generated_at: 2026-05-15T20:02:03.232982+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Work with corpus

Use the corpus module when you need to load, extend, or swap out a source of retrieval entries — `CorpusProtocol` defines the interface, `DirectoryCorpus` loads markdown files from disk, and `AttuneHelpCorpus` wraps the bundled attune-help templates.

## Prerequisites

- Read access to the project source code
- Familiarity with the files listed under [Key files](#key-files)

## Understand the class hierarchy

Before adding or changing anything, orient yourself to the four main classes:

| Class | File | Role |
|---|---|---|
| `CorpusProtocol` | `corpus/base.py` | Interface any corpus must satisfy — exposes `entries()`, `get()`, `name`, and `version` |
| `RetrievalEntry` | `corpus/base.py` | Dataclass for a single template entry; holds `path`, `category`, `content`, `summary`, `related`, `aliases`, and `metadata` |
| `DirectoryCorpus` | `corpus/directory.py` | Loads all `**/*.md` files from a directory root; provides `path_index` and `alias_index`; raises `DuplicateAliasError` when two templates declare the same alias |
| `AttuneHelpCorpus` | `corpus/attune_help.py` | Thin adapter over the bundled attune-help templates; construct with `AttuneHelpCorpus.from_attune_help()` |

## Steps

1. **Choose your entry point.** Decide which corpus class fits your goal:
   - To serve the built-in attune-help templates, call `AttuneHelpCorpus.from_attune_help()`.
   - To load your own markdown directory, instantiate `DirectoryCorpus(root=Path("your/dir"))`.
   - To implement a new corpus source, write a class that satisfies `CorpusProtocol`.

2. **Extend rather than modify base classes.** If you need custom behavior in `DirectoryCorpus` or `AttuneHelpCorpus`, create a subclass and override only the methods you need. Edit a base class directly only when no subclass can satisfy the requirement.

3. **Add or update a `RetrievalEntry` field.** Open `corpus/base.py` and add your field to the `RetrievalEntry` dataclass. Provide a default value so existing callers do not break. If the field must be populated at load time, update the loader in `DirectoryCorpus.entries()` or your custom corpus's `entries()` implementation.

4. **Handle alias collisions.** If your corpus declares aliases, be aware that `DirectoryCorpus` raises `DuplicateAliasError` when two templates share an alias. Catch this error during corpus initialization and resolve the conflict before the corpus is used for retrieval.

5. **Run the tests.** Verify your changes with:
   ```
   pytest -k "corpus"
   ```

## Key files

- `src/attune_rag/corpus/__init__.py` — public exports (`CorpusProtocol`, `DirectoryCorpus`, `RetrievalEntry`, `AliasInfo`, `DuplicateAliasError`)
- `src/attune_rag/corpus/base.py` — `CorpusProtocol`, `RetrievalEntry`, `AliasInfo`, `DuplicateAliasError`
- `src/attune_rag/corpus/directory.py` — `DirectoryCorpus` implementation
- `src/attune_rag/corpus/attune_help.py` — `AttuneHelpCorpus` implementation

## Verify success

After running the test suite, confirm the following:

- `pytest -k "corpus"` exits with no failures.
- Calling `corpus.entries()` on your corpus returns an iterable of `RetrievalEntry` objects with non-empty `path` and `category` fields.
- Calling `corpus.get("some/path")` returns the expected `RetrievalEntry` or `None` — not an exception.
- `corpus.version` returns a stable SHA-256 fingerprint (for `DirectoryCorpus`) or a non-empty string, confirming the corpus loaded successfully.
