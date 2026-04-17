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

## Status

v0.1.0 — initial release. Part of the attune ecosystem
([attune-ai](https://github.com/Smart-AI-Memory/attune-ai),
[attune-help](https://github.com/Smart-AI-Memory/attune-help),
[attune-author](https://github.com/Smart-AI-Memory/attune-author)).

## License

Apache 2.0. See
[LICENSE](https://github.com/Smart-AI-Memory/attune-rag/blob/main/LICENSE).
