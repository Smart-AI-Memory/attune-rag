---
type: task
feature: corpus
depth: task
generated_at: 2026-04-23T03:34:04.286167+00:00
source_hash: fbf3871db9ff126e132e66618572aafec8f5d3bb4da48be33dbd1beb2a75d455
status: generated
---

# Work with corpus

Use the corpus module when you need to load and access document collections for retrieval systems — whether from disk directories, bundled templates, or custom sources.

## Prerequisites

- Access to the project source code
- Understanding of the `RetrievalEntry` data structure
- Familiarity with Python protocols and dataclasses

## Steps

1. **Choose your corpus implementation.**
   Select the corpus type that matches your data source:
   - `DirectoryCorpus` — loads markdown files from a filesystem directory
   - `AttuneHelpCorpus` — provides access to bundled attune-help templates
   - Custom implementation — implement `CorpusProtocol` for other sources

2. **Import the required classes.**
   Add the necessary imports to your module:
   ```python
   from attune_rag.corpus import CorpusProtocol, RetrievalEntry, DirectoryCorpus
   # or
   from attune_rag.corpus import AttuneHelpCorpus
   ```

3. **Initialize your corpus.**
   Create a corpus instance with appropriate configuration:
   ```python
   # For directory-based corpus
   corpus = DirectoryCorpus(
       root=Path("./docs"),
       summaries_file="summaries.json",  # optional
       cross_links_file="links.json",    # optional
       cache=True                        # optional
   )

   # For bundled templates
   corpus = AttuneHelpCorpus()
   ```

4. **Access corpus entries.**
   Use the corpus protocol methods to retrieve documents:
   ```python
   # Get all entries
   for entry in corpus.entries():
       print(f"{entry.path}: {entry.category}")

   # Get specific entry
   entry = corpus.get("path/to/document.md")
   if entry:
       print(entry.content)
   ```

5. **Verify corpus functionality.**
   Check that your corpus loads correctly by examining its metadata and entry count:
   ```python
   print(f"Corpus: {corpus.name} v{corpus.version}")
   entries = list(corpus.entries())
   print(f"Loaded {len(entries)} entries")
   ```

## Verification

Your corpus integration works correctly when:
- `corpus.entries()` returns an iterable of `RetrievalEntry` objects
- `corpus.get(path)` retrieves specific entries by path
- Entry objects contain valid `path`, `category`, and `content` fields
- The corpus `name` and `version` properties return meaningful values

## Key files

- `src/attune_rag/corpus/__init__.py` — public API exports
- `src/attune_rag/corpus/base.py` — protocol definition and data structures
- `src/attune_rag/corpus/directory.py` — filesystem corpus implementation
- `src/attune_rag/corpus/attune_help.py` — bundled templates corpus
