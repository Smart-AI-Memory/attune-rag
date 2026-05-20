---
type: comparison
name: corpus-comparison
feature: corpus
depth: comparison
generated_at: 2026-05-20T03:23:24.680360+00:00
source_hash: 4acdd163679b03efe44559300b991a4426e0e3b739b30950c8f3ec8964e7efc0
status: generated
---

# Comparison: Corpus implementations

## Context

The corpus feature provides a pluggable loading layer built around a single interface, `CorpusProtocol`. Two concrete implementations ship with the library: `DirectoryCorpus`, which loads arbitrary markdown files from disk, and `AttuneHelpCorpus`, which wraps the bundled attune-help templates. Choosing between them comes down to where your content lives and how much control you need over the loading pipeline.

## Feature comparison

| Capability | `DirectoryCorpus` | `AttuneHelpCorpus` |
|---|---|---|
| **Content source** | Any directory of `.md` files (configurable glob) | Bundled attune-help templates only |
| **Entry point** | `DirectoryCorpus(root, ...)` | `AttuneHelpCorpus.from_attune_help()` (class method) |
| **Summaries** | Optional external `summaries_file` or `extra_summaries` dict | Managed by the bundled adapter |
| **Cross-links** | Optional external `cross_links_file` | Managed by the bundled adapter |
| **Alias indexing** | Yes — `alias_index` property; raises `DuplicateAliasError` on collision | Inherited via `CorpusProtocol`; index managed by adapter |
| **Path index** | Yes — `path_index` property (`rel_path → RetrievalEntry`) | `get(path)` lookup only |
| **Corpus version** | Stable SHA-256 fingerprint of loaded content | Delegated to `HelpCorpusAdapter` |
| **Caching** | Configurable (`cache=True` by default) | Handled by adapter |
| **Custom glob** | Yes — override `DEFAULT_GLOB` (`**/*.md`) | No |
| **Typical setup complexity** | Medium — requires a root path and optional sidecar files | Low — single class-method call |

Both implementations satisfy `CorpusProtocol` (`entries()`, `get()`, `name`, `version`), so any code that types against the protocol works with either.

## Tradeoffs

**`DirectoryCorpus` gives you control at the cost of configuration.** You provide the root path and, optionally, sidecar files for summaries and cross-links. In return you get a writable alias index, a full path index for O(1) lookups, SHA-256 content fingerprinting for cache invalidation, and the ability to target any subset of files via a custom glob pattern. If two templates declare the same alias, `DuplicateAliasError` surfaces the conflict immediately (with both paths), so integrity problems do not silently corrupt retrieval.

**`AttuneHelpCorpus` is intentionally narrow.** It is a thin adapter over the bundled attune-help templates — you cannot point it at a different directory, and its indexing behaviour is delegated to `HelpCorpusAdapter`. The upside is zero configuration: `AttuneHelpCorpus.from_attune_help()` is a single call with no required arguments.

## When NOT to use these implementations directly

- If your retrieval logic sits above the corpus layer (for example, in a RAG pipeline or orchestration component), prefer wiring `CorpusProtocol` at the injection point rather than hardcoding a concrete class.
- If you need behaviour that neither implementation exposes — such as remote content sources or incremental updates — implement `CorpusProtocol` in a new class rather than patching internals.
- If you are writing a one-off exploration script, loading markdown files manually is simpler than configuring a full `DirectoryCorpus` instance.

## Use X when…

**Use `AttuneHelpCorpus`** when you need to retrieve from the bundled attune-help templates and have no reason to customise the corpus. It is the right default for code that works exclusively with attune-help content.

**Use `DirectoryCorpus`** when:
- Your templates live outside the attune-help bundle (your own markdown directory, a checked-out docs repo, etc.).
- You need alias deduplication guarantees enforced at load time.
- You need a path index, a custom file glob, or explicit control over summaries and cross-links.
- You want a stable version fingerprint to drive cache invalidation.

`DirectoryCorpus` is the more capable implementation for the majority of real-world use cases. `AttuneHelpCorpus` wins only when your content is exclusively the bundled attune-help templates and you want the simplest possible setup.

## Source files

- `src/attune_rag/corpus/__init__.py`
- `src/attune_rag/corpus/base.py`
- `src/attune_rag/corpus/directory.py`
- `src/attune_rag/corpus/attune_help.py`

**Tags:** `corpus`, `loader`, `markdown`, `attune-help`
