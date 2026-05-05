# attune-rag

Lightweight, LLM-agnostic RAG pipeline with pluggable
corpora. Works with Claude, OpenAI, Gemini, or any LLM.

- **No LLM SDK at install time.** All provider deps are
  optional extras.
- **Pluggable corpus.** Use attune-help (the default), any
  markdown directory, or your own `CorpusProtocol`.
- **Returns a prompt string** by default — send it to
  whatever LLM you like. Optional provider adapters ship
  convenience wrappers.
- **Optional hybrid retrieval.** `QueryExpander` and
  `LLMReranker` layer Claude Haiku on top of keyword
  retrieval to improve recall and precision — both opt-in,
  both fail-safe.

## Install

```bash
pip install attune-rag                     # core only
pip install 'attune-rag[attune-help]'      # + bundled help corpus
pip install 'attune-rag[claude]'           # + Claude adapter
pip install 'attune-rag[openai]'           # + OpenAI adapter
pip install 'attune-rag[gemini]'           # + Gemini adapter
pip install 'attune-rag[all]'              # everything
```

## Quick start — Claude

```bash
pip install 'attune-rag[attune-help,claude]'
```

```python
import asyncio
from attune_rag import RagPipeline

async def main():
    pipeline = RagPipeline()  # defaults to AttuneHelpCorpus
    response, result = await pipeline.run_and_generate(
        "How do I run a security audit with attune?",
        provider="claude",
    )
    print(response)
    print("\nSources:", [h.entry.path for h in result.citation.hits])

asyncio.run(main())
```

## Quick start — OpenAI

```bash
pip install 'attune-rag[attune-help,openai]'
```

```python
response, result = await pipeline.run_and_generate(
    "...", provider="openai", model="gpt-4o",
)
```

## Quick start — Gemini

```bash
pip install 'attune-rag[attune-help,gemini]'
```

```python
response, result = await pipeline.run_and_generate(
    "...", provider="gemini", model="gemini-1.5-pro",
)
```

## Quick start — custom corpus, any LLM

```python
from pathlib import Path
from attune_rag import RagPipeline, DirectoryCorpus

pipeline = RagPipeline(corpus=DirectoryCorpus(Path("./my-docs")))
result = pipeline.run("How do I...?")

# Send result.augmented_prompt to whatever LLM you use.
# The pipeline itself does NOT call an LLM unless you use
# run_and_generate or call a provider adapter yourself.
```

## Hybrid retrieval (optional)

`QueryExpander` and `LLMReranker` require the `[claude]` extra and an
`ANTHROPIC_API_KEY`. Both are opt-in and fail-safe — any API error
falls back to keyword-only order automatically.

```python
from attune_rag import RagPipeline, LLMReranker, QueryExpander

# Reranker only (recommended for precision):
pipeline = RagPipeline(reranker=LLMReranker())

# Expander + reranker (max coverage):
pipeline = RagPipeline(
    expander=QueryExpander(),
    reranker=LLMReranker(),
)
```

## Template editor primitives (`attune_rag.editor`)

Headless toolkit for tools that need to validate, lint, and refactor a
template corpus — used by the [`attune-gui`](https://pypi.org/project/attune-gui/)
template editor and the [`attune-author`](https://pypi.org/project/attune-author/)
`edit` CLI, but works standalone with any
[`CorpusProtocol`](https://github.com/Smart-AI-Memory/attune-rag).

| API | What it does |
|-----|---------------|
| `load_schema()` | Loads `template_schema.json` (the v1 frontmatter contract: required `type` enum + `name`; optional `tags`, `aliases`, `summary`, `source`, `hash`; `additionalProperties: true`). |
| `parse_frontmatter(text)` / `validate_frontmatter(data)` | Split a template into frontmatter + body and report typed `FrontmatterIssue`s — used by linters and editors. |
| `lint_template(text, rel_path, corpus)` | Returns `Diagnostic[]` for schema violations, broken `[[alias]]` references, and depth-marker sequence errors. 1-indexed line/col ranges. |
| `autocomplete_tags(corpus, prefix, limit)` / `autocomplete_aliases(corpus, prefix, limit)` | Prefix-match completions ranked by frequency (tags) or lexical proximity (aliases). Sub-ms on 1k templates. |
| `find_references(corpus, name, kind)` | Locate every alias/tag/path occurrence across body, frontmatter, and `cross_links.json`. |
| `plan_rename(corpus, old, new, kind)` | Build a `RenamePlan` (one `FileEdit` per affected file with unified-diff hunks) for `kind="alias"` or `"tag"`. Raises `RenameCollisionError` on existing alias targets. |
| `apply_rename(corpus, plan)` | Atomically apply the plan (tempfile-per-file + sequential rename + drift-detection rollback). Returns the list of affected paths. |

Schema, lint, and rename are pure functions over `CorpusProtocol` — no I/O,
no global state. All three pieces are tested as a unit and used live by the
attune-gui editor's `/api/corpus/<id>/lint`, `/autocomplete`, and
`/refactor/rename/{preview,apply}` routes.

```python
from attune_rag import DirectoryCorpus
from attune_rag.editor import lint_template, plan_rename, apply_rename

corpus = DirectoryCorpus(Path("./templates")).load()

# Validate a template before saving
diagnostics = lint_template(
    text=Path("./templates/concepts/foo.md").read_text(),
    rel_path="concepts/foo.md",
    corpus=corpus,
)

# Rename an alias across the whole corpus
plan = plan_rename(corpus, old="oldname", new="newname", kind="alias")
print(f"Affects {len(plan.edits)} files")
affected = apply_rename(corpus, plan)
```

## Dashboard

```bash
attune-rag dashboard show    # live terminal dashboard
attune-rag dashboard render --out report.html  # HTML snapshot
```

## Roadmap — embeddings (next minor release)

Keyword retrieval + optional Claude reranker currently carry
attune-rag past 87% P@1 on the attune-help golden set. The
remaining misses are queries with zero token overlap against
their target doc (e.g. "vulnerability scan" →
`tool-security-audit.md`). Closing that gap needs vector search.

Next minor release will ship `attune-rag[embeddings]` using
[`fastembed`](https://github.com/qdrant/fastembed) for local,
CPU-only embeddings — no new network dependency, no API key
required at retrieval time. Keyword retrieval stays the default;
embeddings layer in opt-in, same shape as `QueryExpander` and
`LLMReranker`.

See
[CHANGELOG.md](https://github.com/Smart-AI-Memory/attune-rag/blob/main/CHANGELOG.md)
for the decision record and remaining-gap analysis.

## Prompt caching (Claude only)

When using the Claude provider, `run_and_generate` automatically enables
[Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
on the stable RAG context prefix (≥ 1 024 chars). This eliminates
repeated token costs on the corpus portion of the prompt when the same
context block is reused across calls.

No configuration needed — the provider handles the `cache_control`
header automatically.

## Status

v0.1.10. Part of the attune ecosystem
([attune-ai](https://github.com/Smart-AI-Memory/attune-ai),
[attune-help](https://github.com/Smart-AI-Memory/attune-help),
[attune-author](https://github.com/Smart-AI-Memory/attune-author)).

## License

Apache 2.0. See
[LICENSE](https://github.com/Smart-AI-Memory/attune-rag/blob/main/LICENSE).
