---
type: task
name: cli-task
feature: cli
depth: task
generated_at: 2026-06-10T06:04:14.432466+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Work with the attune-rag CLI

Use the attune-rag CLI when you want to run retrieval-augmented queries, inspect your corpus, or check available LLM providers from the command line without writing Python code.

## Prerequisites

- The `attune_rag` package installed in your environment
- A corpus directory of markdown files if you plan to run queries or inspect corpus statistics

## Steps

1. **Run a RAG query against your corpus.**
   Call `attune-rag query` with the required question and point it at your markdown directory:

   ```sh
   attune-rag query "What is retrieval-augmented generation?" \
     --corpus-path ./docs
   ```

   The command prints a grounded answer with citations.

2. **Select a retrieval strategy.**
   Pass `--retriever` to choose how documents are ranked. The accepted values are `keyword`, `hybrid`, and `transformer`:

   ```sh
   attune-rag query "Your question" \
     --corpus-path ./docs \
     --retriever hybrid
   ```

3. **Set the keyword abstention threshold.**
   Use `--min-score` to control the minimum relevance score below which the keyword retriever abstains from returning a result:

   ```sh
   attune-rag query "Your question" \
     --corpus-path ./docs \
     --min-score 0.4
   ```

4. **Choose a prompt template.**
   Use `--prompt-variant` to select the prompt template used when generating the answer:

   ```sh
   attune-rag query "Your question" \
     --corpus-path ./docs \
     --prompt-variant concise
   ```

5. **Inspect corpus statistics.**
   Run `attune-rag corpus-info` to see a summary of your corpus. This command also accepts `--corpus-path`:

   ```sh
   attune-rag corpus-info --corpus-path ./docs
   ```

6. **List available LLM providers.**
   Run `attune-rag providers` to see which LLM providers have their required extras installed:

   ```sh
   attune-rag providers
   ```

7. **Handle setup errors.**
   If you pass a bad path, conflicting flags, or a provider whose extras are not installed, the CLI prints a single actionable error message and exits with code `2`. Check the message, correct the flag or install the missing extras, then rerun your command.

## Verify success

- `attune-rag query` prints an answer followed by source citations — no traceback.
- `attune-rag corpus-info` prints document counts and corpus statistics.
- `attune-rag providers` lists at least one available provider.
- Any misconfigured invocation exits with code `2` and a one-line message describing the problem.

## Key files

- `src/attune_rag/cli.py` — defines `build_parser()` and `main(argv)`
