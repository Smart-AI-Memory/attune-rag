---
type: note
name: benchmark-note
feature: benchmark
depth: note
generated_at: 2026-05-20T03:30:01.600493+00:00
source_hash: 82975cf88c844b87657deb87845f45f4f5fbc32319ccba10e0eb8a798867630f
status: generated
---

# Note: benchmark

## Context

The `benchmark` module (`src/attune_rag/benchmark.py`) is a retrieval and optional faithfulness benchmark runner. It is designed to gate CI pipelines on configurable quality thresholds, measuring precision and recall against a query file you supply.

Faithfulness scoring is opt-in and enabled with the `--with-faithfulness` flag. Without it, the runner evaluates retrieval quality only.

## How it works

The module is function-first: there are no classes to instantiate. The entry point is `main()`, which accepts an argument list and returns `0` on success. You invoke it directly — either from the command line or by passing a list of arguments programmatically.

```python
from attune_rag.benchmark import main

exit_code = main(["--with-faithfulness"])
```

Configurable thresholds and the query file path are passed as arguments to `main()`; the runner exits non-zero if any threshold is not met, making it suitable as a CI gate.

**Tags:** `benchmark`, `ci`, `precision`, `recall`, `quality`
