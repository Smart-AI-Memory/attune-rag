---
type: quickstart
name: benchmark-quickstart
feature: benchmark
depth: quickstart
generated_at: 2026-05-20T03:30:01.595586+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Benchmark runner quickstart

Run a retrieval benchmark — with optional faithfulness scoring — that gates CI on configurable thresholds.

```python
from attune_rag.benchmark import main

exit_code = main()
print(exit_code)  # 0
```

## Prerequisites

- The project is cloned and installed locally.
- You have a query file ready if you want to test against custom queries.

## Run your first benchmark

1. **Import and call `main`.** With no arguments, the runner uses its defaults:

   ```python
   from attune_rag.benchmark import main

   exit_code = main()
   ```

   A return value of `0` means all metrics passed their thresholds.

2. **Pass a custom query file.** Supply your own queries via `argv`:

   ```python
   exit_code = main(["--queries", "my_queries.jsonl"])
   ```

3. **Enable faithfulness scoring.** Add `--with-faithfulness` to score generated answers against their source passages:

   ```python
   exit_code = main(["--queries", "my_queries.jsonl", "--with-faithfulness"])
   ```

   A non-zero return value means at least one metric fell below its configured threshold — check the logged output to see which one.

## Expected output

```
Retrieval precision: 0.91  ✓
Retrieval recall:    0.87  ✓
Faithfulness:        0.93  ✓
Benchmark passed (exit 0)
```

**Next:** Wire `main()` into your CI pipeline and set threshold values in your benchmark configuration to enforce quality gates on every pull request.

## Source files

- `src/attune_rag/benchmark.py`

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
