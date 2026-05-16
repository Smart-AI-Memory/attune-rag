---
feature: benchmark
depth: task
generated_at: 2026-05-16T08:44:05.216307+00:00
source_hash: 3ecf192b8459343df49cfb4a43d472af419ebfe935b46e8c0f34d5635079e14d
status: generated
---

# Work with benchmark

Use benchmark when you need to precision/recall/faithfulness benchmark runner — gates ci on configurable thresholds; supports custom query files and optional faithfulness scoring via --with-faithfulness.

## Prerequisites

- Access to the project source code
- Familiarity with the files under src/attune_rag/benchmark.py

## Steps

1. **Understand the current behavior.**
   Read the entry points to see what benchmark
   does today before making changes.
   The primary functions are:
   - `main()` in `src/attune_rag/benchmark.py`
2. **Locate the right function to change.**
   Each function has a single responsibility. Read its
   docstring, parameters, and return type to confirm it
   owns the behavior you need to modify.

3. **Make your change.**
   Follow existing patterns in the file — naming
   conventions, error handling style, and logging.

4. **Run the related tests.**
   This catches regressions before they reach other
   developers. Target with `pytest -k "benchmark"`.

## Key files

- `src/attune_rag/benchmark.py`

## Common modifications

Functions you are most likely to modify:

- `main()` in `src/attune_rag/benchmark.py`
