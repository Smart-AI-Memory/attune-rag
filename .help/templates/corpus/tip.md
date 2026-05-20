---
type: tip
name: corpus-tip
feature: corpus
depth: tip
generated_at: 2026-05-20T03:23:24.675574+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Tip: Use `aliases` and `related` on `RetrievalEntry` to surface templates under multiple names

## Recommendation

When you add a `RetrievalEntry` to your corpus, populate its `aliases` and `related` tuples — even if you only have one alias today. `DirectoryCorpus` indexes every alias into `alias_index` at load time, so lookups by alternate name are instant without any extra work on your part.

**Why:** A template that can only be found by its exact path is half as discoverable as one that can be found by the terms your users actually type.

**Tradeoff:** Every alias you declare must be globally unique across the corpus. If two templates claim the same alias, `DirectoryCorpus` raises `DuplicateAliasError` at load time. Audit your alias names before adding them, especially when merging two corpora.

## Key types

- `RetrievalEntry` — dataclass holding `path`, `category`, `content`, `summary`, `related`, `aliases`, and `metadata`. The fields `aliases` and `related` both default to empty tuples, so you will not see a load error if you omit them — but you will miss the lookup benefit.
- `DirectoryCorpus.alias_index` — `dict[str, AliasInfo]` built from every alias declared in the loaded corpus. Use it to verify that an alias resolves to the entry you expect.
- `DuplicateAliasError` — raised with the conflicting alias string and both `first_path` / `second_path`, so you can identify the collision immediately.

## Source files

- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
