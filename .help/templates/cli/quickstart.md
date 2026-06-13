---
type: quickstart
name: cli-quickstart
feature: cli
depth: quickstart
generated_at: 2026-06-10T06:07:13.455629+00:00
source_hash: 96db3d6bf557349fb1cbc8ae947bdd3fa30475c1926eb4172b0875e533ece578
status: generated
---

# Quickstart: attune-rag CLI

Run your first retrieval-augmented query against a local markdown corpus:

```bash
attune-rag query "What is the refund policy?" --corpus-path ./docs
```

You should see a grounded answer followed by citations drawn from your markdown files. If extras are missing or the path is wrong, the CLI prints a one-line actionable message and exits with code 2 instead of a traceback.

## Step 1: Point the CLI at your corpus

Pass `--corpus-path` to tell the CLI where your markdown files live:

```bash
attune-rag query "Your question here" --corpus-path ./docs
```

## Step 2: Choose a retriever

Use `--retriever` to select how documents are retrieved. The accepted values are `keyword`, `hybrid`, and `transformer`:

```bash
attune-rag query "Your question here" \
  --corpus-path ./docs \
  --retriever hybrid
```

Use `--min-score` to set the keyword abstention threshold if you want to suppress low-confidence results.

## Step 3: Inspect your corpus

Verify the CLI can see your content before running queries:

```bash
attune-rag corpus-info --corpus-path ./docs
```

Expected output: a summary of document count, token statistics, and any indexing warnings.

## Step 4: Check available providers

Confirm which LLM providers are ready to use:

```bash
attune-rag providers
```

Expected output: a list of providers whose optional extras are installed. If a provider is missing, install the corresponding extras package and re-run.

---

**Next:** Swap in `--retriever transformer` and compare answer quality against `--retriever keyword` to find the retrieval setting that works best for your corpus.
