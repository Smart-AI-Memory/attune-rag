---
type: quickstart
name: cli-quickstart
feature: cli
depth: quickstart
generated_at: 2026-05-20T03:30:50.403001+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e523ece578
status: generated
---

# Quickstart: cli

Run a RAG query from the command line and see a grounded answer with citations in seconds.

```bash
attune-rag query "What is the return policy?"
```

## Prerequisites

- The project is cloned and installed locally (`pip install -e .` from the repo root)

## Steps

1. **Run a query.** Pass your question as a positional argument:

   ```bash
   attune-rag query "What is the return policy?"
   ```

   You should see output similar to:

   ```
   Answer: Returns are accepted within 30 days of purchase. [1]

   Citations:
     [1] docs/policies.md, line 12
   ```

2. **Check corpus statistics.** Confirm your corpus loaded correctly:

   ```bash
   attune-rag corpus-info
   ```

   Expected output:

   ```
   Documents indexed: 42
   Chunks:            317
   Embedding model:   text-embedding-ada-002
   ```

3. **Explore available options.** Print the full help text to see every flag:

   ```bash
   attune-rag --help
   ```

## Next

Read the `build_parser()` reference in `src/attune_rag/cli.py` to learn how to extend the CLI with custom subcommands.

**Tags:** `cli`, `query`, `corpus-info`
