---
type: note
name: dashboard-note
feature: dashboard
depth: note
generated_at: 2026-05-20T03:33:38.815026+00:00
source_hash: 48be0a4fd811c784bc44e073b2ac5906c205487b317ef813d32ca7c5e3b936cc
status: generated
---

# Note: dashboard

## Context

The dashboard feature provides a living-docs view of a registered corpus through a three-stage pipeline:

1. **Refresh** — `build_snapshot()` runs the benchmark against the corpus and returns a snapshot dict. If `queries.yaml` is missing, it returns a partial snapshot that includes an error field rather than failing outright.
2. **Render** — `render()` writes a self-contained HTML report to a file path you specify, with the snapshot embedded as JSON between the `__ATTUNE_SNAPSHOT__` and `__ATTUNE_TITLE__` sentinels in the template.
3. **Show** — `display()` pretty-prints the snapshot to the terminal using Rich.

Each stage has a dedicated source file and its own CLI entry point under `attune-rag dashboard`.

## Content

The dashboard module exposes top-level functions rather than classes. You call them directly without instantiating anything first.

| Function | File | Purpose |
|---|---|---|
| `build_snapshot(corpus_package, queries_path)` | `dashboard/refresh.py` | Returns a snapshot dict; returns a partial dict with an error key if `queries.yaml` is absent |
| `render(out, snapshot, title)` | `dashboard/render.py` | Renders the dashboard HTML template to `out` with `snapshot` embedded as JSON; returns the resolved output path |
| `display(snapshot, console)` | `dashboard/show.py` | Prints the snapshot to the terminal via Rich; accepts an optional `Console` instance for testing or redirection |

The refresh and show stages each also expose a `main()` function (return code `0` on success) that wires the stage into the CLI.

Two template sentinels — `_SENTINEL_SNAPSHOT` (`__ATTUNE_SNAPSHOT__`) and `_SENTINEL_TITLE` (`__ATTUNE_TITLE__`) — mark the injection points in the HTML template. They are internal constants and are not part of the public API.

## Source files

- `src/attune_rag/dashboard/__init__.py`
- `src/attune_rag/dashboard/refresh.py`
- `src/attune_rag/dashboard/render.py`
- `src/attune_rag/dashboard/show.py`

**Tags:** `dashboard`, `living-docs`, `html`, `terminal`, `snapshot`, `freshness`
