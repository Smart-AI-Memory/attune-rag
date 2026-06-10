---
type: faq
name: cli-faq
feature: cli
depth: faq
generated_at: 2026-06-10T06:07:13.453328+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# CLI FAQ

## What does the CLI do?

`attune-rag` is the command-line entry point for debugging retrieval. It exposes three subcommands:

- **`attune-rag query`** — runs a RAG query and prints the grounded answer with citations.
- **`attune-rag corpus-info`** — shows corpus statistics.
- **`attune-rag providers`** — lists LLM providers whose extras are installed.

Setup errors (missing extras, bad paths, conflicting flags) print a one-line actionable message and exit with code `2` instead of a traceback.

## How do I point the CLI at my own content?

Pass `--corpus-path` to either `attune-rag query` or `attune-rag corpus-info` with the path to your markdown directory.

## How do I choose a retrieval strategy?

Use `--retriever {keyword,hybrid,transformer}` with `attune-rag query` to select a retrieval method. The three values correspond to an opt-in retrieval ladder: `keyword` is the baseline, `hybrid` combines keyword and semantic signals, and `transformer` uses a transformer-based retriever.

## What does `--min-score` do?

It sets the keyword abstention threshold for `attune-rag query`. Queries whose best match falls below this score are suppressed rather than returned as low-confidence answers.

## How do I change the prompt template?

Pass `--prompt-variant` to `attune-rag query` to select the prompt template used when generating the answer.

## What are the main entry points in code?

The public API in `attune_rag.cli` has two functions:

- `build_parser() -> argparse.ArgumentParser` — constructs the argument parser for all subcommands.
- `main(argv: list[str] | None = None) -> int` — parses arguments and runs the selected subcommand. Pass a list of strings to call it programmatically, or pass `None` to read from `sys.argv`.

## Can I call the CLI programmatically?

Yes. Import `main` from `attune_rag.cli` and pass a list of argument strings: `main(["query", "--corpus-path", "docs/", "your question"])`. It returns an integer exit code.

## What exit code does the CLI return on error?

It exits with code `2` when it encounters a setup error such as a missing extras package, a bad corpus path, or conflicting flags. Valid runs return `0`.

## Where is the source?

`src/attune_rag/cli.py`

---

**Tags:** `cli`, `query`, `corpus-info`, `corpus-path`, `retriever-tiers`, `abstention`
