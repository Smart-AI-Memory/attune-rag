---
type: quickstart
name: benchmark-quickstart
feature: benchmark
depth: quickstart
generated_at: 2026-06-10T06:07:59.724934+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Quickstart: Run Your First Retrieval Benchmark

Run `attune-rag-benchmark` from your terminal to benchmark retrieval quality and get a pass/fail result you can wire into CI.

```sh
attune-rag-benchmark
```

Expected output: the benchmark completes and exits `0` when all configured thresholds pass.

## Steps

**1. Install the package**

Make sure `attune-rag` is installed in your local environment. The benchmark entry point is registered automatically — no separate install step needed.

**2. Choose a retriever tier**

Pass `--retriever` to target a specific tier:

```sh
attune-rag-benchmark --retriever hybrid
```

Valid values are `keyword`, `hybrid`, and `transformer`. If the selected tier's optional dependency is missing, the command exits `2` and prints an install hint.

**3. Add faithfulness scoring (optional)**

To score faithfulness in addition to precision and recall, add `--with-faithfulness`:

```sh
attune-rag-benchmark --retriever hybrid --with-faithfulness
```

**4. Calibrate the abstention threshold (optional)**

Run with `--calibrate-abstention` to tune the threshold before committing results to CI:

```sh
attune-rag-benchmark --calibrate-abstention
```

## What you just did

You ran `attune-rag-benchmark`, selected a retriever tier, and confirmed the benchmark exits `0` on a passing run. The same command — with an exit-code check — is all you need to gate a CI pipeline on retrieval quality.

## Next:

Supply a custom query file to benchmark against your own data — check `attune-rag-benchmark --help` for the flag that points to your query file path.
