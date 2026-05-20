---
type: tip
name: dashboard-tip
feature: dashboard
depth: tip
generated_at: 2026-05-20T03:33:38.813119+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Tip: Call `build_snapshot()` before `render()` or `display()`

The dashboard pipeline has three discrete stages — refresh, render, and show — and each stage owns exactly one responsibility. Skipping or reordering stages is the most common source of stale or broken output.

**Why it matters:** `render()` and `display()` both accept a `dict[str, Any]` snapshot, but they do not validate it. Passing a stale or partial snapshot produces silently wrong output, not an error.

## The recommended sequence

1. Call `build_snapshot(corpus_package, queries_path)` to produce a fresh snapshot dict. If `queries.yaml` is missing, the function returns a partial snapshot that includes an error key — check for that before proceeding.
2. Pass the snapshot to `render(out, snapshot)` to write a self-contained HTML file with the snapshot embedded as JSON, or to `display(snapshot)` to pretty-print it in the terminal via Rich. These two are independent; you can call either or both.

```python
from attune_rag.dashboard.refresh import build_snapshot
from attune_rag.dashboard.render import render
from attune_rag.dashboard.show import display
from pathlib import Path

snapshot = build_snapshot()          # raises nothing; check for error key
if "error" not in snapshot:
    render(Path("report.html"), snapshot)
    display(snapshot)
```

## Tradeoff

Calling `build_snapshot()` reruns the benchmark against the full corpus every time, which can be slow on large corpora. If you only need to re-render or re-display existing results, cache the snapshot dict and pass it directly to `render()` or `display()` — but accept that the output may not reflect the current state of the corpus.

## What to avoid

Anything prefixed with an underscore — `_SENTINEL_SNAPSHOT`, `_SENTINEL_TITLE`, `_SYSTEM_DIRS` — is an implementation detail of the render pipeline. Do not reference these constants directly; they can change without notice.

## Source files

- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`
- `src/attune_rag/dashboard/__init__.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
